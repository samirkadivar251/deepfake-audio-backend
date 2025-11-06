# deepfake_api.py
import os
import re
import io
import glob
import tempfile
from typing import Tuple

import numpy as np
from PIL import Image
import librosa
import soundfile as sf

import torch
import torch.nn as nn
from torchvision import models, transforms

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# =======================
# CONFIG - adjust if needed
# =======================
CHECKPOINT_DIR = os.path.join(os.getcwd(), "checkpoints")  # where your .pth files are
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# IMPORTANT: set these to match how you organized folders when training with ImageFolder.
# ImageFolder sorts names alphabetically; if your train folders were named `fake` and `real`
# then classes = ['fake', 'real'] -> index 0=Fake, 1=Real.
CLASS_NAMES = ['fake', 'real']  # <-- change if necessary

# ImageNet normalization used in training
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMG_SIZE = (224, 224)

# Model architecture helper (must match training)
def create_model(num_classes: int = 2) -> nn.Module:
    model = models.densenet121(pretrained=True)
    # freeze pretrained layers (same as training)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    return model

# =======================
# Utils: checkpoint loading
# =======================
def get_latest_checkpoint(checkpoint_dir: str = CHECKPOINT_DIR) -> str:
    if not os.path.isdir(checkpoint_dir):
        return None
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if not checkpoints:
        return None
    # Attempt to sort by numeric epoch if possible, fallback to modified time
    def keyfn(x):
        nums = re.findall(r'\d+', x)
        return int(nums[-1]) if nums else os.path.getmtime(os.path.join(checkpoint_dir, x))
    checkpoints.sort(key=keyfn)
    return os.path.join(checkpoint_dir, checkpoints[-1])

# =======================
# Audio -> Mel-spectrogram -> PIL RGB Image
# =======================
def audio_to_mel_image(file_path: str, sr_target: int = 22050, n_mels: int = 128) -> Image.Image:
    """
    Load audio file and convert to a 3-channel PIL image (RGB) of mel-spectrogram.
    """
    # librosa.load will handle wav and mp3 (if ffmpeg / audioread backend available)
    y, sr = librosa.load(file_path, sr=sr_target, mono=True)
    if y.size == 0:
        raise ValueError("Empty audio file")

    # Optionally pad or trim to a fixed duration for stable input (not required but helpful)
    # Here we keep full length; mel-spectrogram will be computed on full audio.
    # Compute mel-spectrogram (power)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)  # in dB

    # Normalize dB to 0-255 for image conversion
    S_min, S_max = S_db.min(), S_db.max()
    S_norm = 255 * (S_db - S_min) / (S_max - S_min + 1e-6)
    S_uint8 = S_norm.astype(np.uint8)

    # Convert to PIL image (grayscale) then to RGB (3 channels)
    img = Image.fromarray(S_uint8)
    img = img.convert("RGB")  # replicate to 3 channels

    # Resize to model's expected size
    img = img.resize(IMG_SIZE, Image.BILINEAR)
    return img

# =======================
# Preprocessing (same transforms used in training)
# =======================
preprocess = transforms.Compose([
    transforms.ToTensor(),  # converts PIL image (H x W x C) in range [0,255] to float tensor [0,1]
    transforms.Normalize(MEAN, STD),
])

# =======================
# Model loading
# =======================
print("Loading model...")
model = create_model(num_classes=len(CLASS_NAMES))
latest_ckpt = get_latest_checkpoint()
if latest_ckpt is None:
    print("WARNING: No checkpoint found. Please place your .pth files in the `checkpoints` folder.")
else:
    print(f"Loading weights from: {latest_ckpt}")
    state = torch.load(latest_ckpt, map_location=DEVICE)
    # If the checkpoint uses keys like 'model_state_dict' adapt accordingly:
    if isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
model = model.to(DEVICE)
model.eval()
print("Model ready on device:", DEVICE)

# =======================
# FastAPI app
# =======================
app = FastAPI(title="Deepfake Audio Detection API",
              description="Upload an audio file (wav/mp3) and get back Real/Fake + confidence.",
              version="1.0")

# Allow CORS (so your Flutter app can call this)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production lock this down to your app origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "ok", "message": "Deepfake Audio Detection API. POST /predict with file field 'file'."}

def predict_from_image(pil_img: Image.Image) -> Tuple[str, float, dict]:
    """
    Run model on PIL image and return (label, confidence_percent, probs_dict)
    """
    input_tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)  # shape: (1,3,H,W)
    with torch.no_grad():
        outputs = model(input_tensor)
        # for classification, apply softmax
        probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy().squeeze()
    # probs is array length num_classes
    top_idx = int(np.argmax(probs))
    label = CLASS_NAMES[top_idx] if CLASS_NAMES else str(top_idx)
    confidence = float(probs[top_idx] * 100.0)
    probs_dict = {CLASS_NAMES[i]: float(probs[i] * 100.0) for i in range(len(probs))}
    return label, confidence, probs_dict

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an uploaded audio file (wav or mp3). Returns JSON with label and confidence.
    """
    # Basic extension check
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
        raise HTTPException(status_code=400, detail="Unsupported file type. Allowed: wav, mp3, flac, ogg, m4a")

    # Save to a temp file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        # Convert audio -> mel image
        try:
            pil_img = audio_to_mel_image(tmp_path)
        except Exception as e:
            # try using soundfile as fallback to ensure readable format
            try:
                data, sr = sf.read(tmp_path)
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                # write temporary wav and re-run
                wav_tmp = tmp_path + ".wav"
                sf.write(wav_tmp, data, sr)
                pil_img = audio_to_mel_image(wav_tmp)
                os.remove(wav_tmp)
            except Exception as e2:
                raise HTTPException(status_code=500, detail=f"Error processing audio file: {e} | fallback error: {e2}")

        # Predict
        label, confidence, probs = predict_from_image(pil_img)

        # Clean up temp file
        try:
            os.remove(tmp_path)
        except Exception:
            pass

        return JSONResponse({
            "label": label,
            "confidence": round(confidence, 2),
            "probabilities": {k: round(v, 2) for k, v in probs.items()}
        })

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Server error: {exc}")

# Optionally run with: python deepfake_api.py
if __name__ == "__main__":
    # Use uvicorn with more workers in production / on Render use the recommended command
    uvicorn.run("deepfake_api:app", host="0.0.0.0", port=8000, reload=True)
