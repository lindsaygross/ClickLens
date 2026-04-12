"""
ClickLens FastAPI Backend
Serves EfficientNet-B0 thumbnail CTR predictions and Grad-CAM explanations.
"""

import base64
import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clicklens")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_LABELS = {0: "Low", 1: "Medium", 2: "High"}
MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "efficientnet_best.pth"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
device: torch.device = torch.device("cpu")
model: nn.Module | None = None


def _select_device() -> torch.device:
    """Pick the best available accelerator."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_model(dev: torch.device, weights_path: Path) -> nn.Module:
    """Construct EfficientNet-B0 with custom head and load weights."""
    net = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
    num_features = net.num_features  # 1280
    net.classifier = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 3),
    )
    state_dict = torch.load(str(weights_path), map_location=dev)
    net.load_state_dict(state_dict)
    net.to(dev)
    net.eval()
    return net


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global device, model
    device = _select_device()
    logger.info("Using device: %s", device)

    if MODEL_PATH.exists():
        try:
            model = _build_model(device, MODEL_PATH)
            logger.info("Model loaded from %s", MODEL_PATH)
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            model = None
    else:
        logger.warning("Model file not found at %s – starting without model", MODEL_PATH)
        model = None

    yield  # application is running

    # Cleanup (nothing specific needed)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ClickLens API",
    description="Thumbnail CTR prediction and Grad-CAM explanations",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Mock predictions (used when model is not loaded)
# ---------------------------------------------------------------------------
import random

def _mock_predict(filename: str) -> dict:
    """Return randomised but realistic-looking predictions for demo/dev use."""
    rng = random.Random(filename)  # seed by filename so results are stable per file
    raw = [rng.uniform(0.05, 0.9) for _ in range(3)]
    total = sum(raw)
    probs = [round(v / total, 4) for v in raw]
    # Adjust so they sum to exactly 1.0 after rounding
    probs[2] = round(1.0 - probs[0] - probs[1], 4)
    scores = {CLASS_LABELS[i]: probs[i] for i in range(3)}
    pred_idx = int(max(range(3), key=lambda i: probs[i]))
    return {
        "predicted_class": CLASS_LABELS[pred_idx],
        "confidence": probs[pred_idx],
        "scores": scores,
        "mock": True,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ensure_model():
    """Raise 503 if the model is not loaded."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")


def _load_image(raw_bytes: bytes) -> Image.Image:
    """Open raw bytes as an RGB PIL image."""
    return Image.open(io.BytesIO(raw_bytes)).convert("RGB")


def _predict(img: Image.Image) -> dict:
    """Run a single image through the model and return scores dict + predicted class."""
    tensor = PREPROCESS(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    scores = {CLASS_LABELS[i]: round(float(probs[i]), 4) for i in range(3)}
    pred_idx = int(np.argmax(probs))
    return {
        "predicted_class": CLASS_LABELS[pred_idx],
        "confidence": round(float(probs[pred_idx]), 4),
        "scores": scores,
    }


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
def _gradcam(img: Image.Image) -> dict:
    """
    Compute Grad-CAM for the predicted class using model.conv_head as target
    layer.  Returns base64 JPEG of the heatmap overlay plus prediction info.
    """
    # Prepare tensor
    tensor = PREPROCESS(img).unsqueeze(0).to(device).requires_grad_(False)
    input_tensor = tensor.clone().detach().requires_grad_(True)

    # Hooks to capture activations and gradients
    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    target_layer = model.conv_head
    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        # Forward
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)
        pred_idx = int(torch.argmax(probs))

        # Backward for predicted class
        model.zero_grad()
        logits[0, pred_idx].backward()

        # Compute Grad-CAM weights
        grads = gradients[0].squeeze(0)   # (C, H, W)
        acts = activations[0].squeeze(0)  # (C, H, W)
        weights = grads.mean(dim=(1, 2))  # (C,)

        cam = torch.zeros(acts.shape[1:], device=device)
        for i, w in enumerate(weights):
            cam += w * acts[i]
        cam = F.relu(cam)

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        cam_np = cam.cpu().numpy()
    finally:
        fwd_handle.remove()
        bwd_handle.remove()

    # Resize CAM to original image size
    img_np = np.array(img.convert("RGB"))
    h, w = img_np.shape[:2]
    cam_resized = cv2.resize(cam_np, (w, h))

    # Create heatmap overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = np.uint8(0.6 * img_np + 0.4 * heatmap)

    # Encode to base64 JPEG
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", overlay_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    probs_np = probs.detach().cpu().numpy()
    scores = {CLASS_LABELS[i]: round(float(probs_np[i]), 4) for i in range(3)}

    return {
        "heatmap": b64,
        "predicted_class": CLASS_LABELS[pred_idx],
        "confidence": round(float(probs_np[pred_idx]), 4),
        "scores": scores,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    if len(files) < 1 or len(files) > 4:
        raise HTTPException(status_code=400, detail="Upload between 1 and 4 images")

    results = []
    for upload in files:
        raw = await upload.read()

        if model is None:
            # Demo mode: return mock predictions so the UI works without a trained model
            info = _mock_predict(upload.filename or f"file_{len(results)}")
        else:
            try:
                img = _load_image(raw)
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not open image: {upload.filename}",
                )
            info = _predict(img)

        info["filename"] = upload.filename
        results.append(info)

    # Rank by highest "High" probability (descending)
    results.sort(key=lambda r: r["scores"]["High"], reverse=True)
    for rank, r in enumerate(results, start=1):
        r["rank"] = rank

    is_mock = model is None
    return {"results": results, "mock": is_mock}


@app.post("/gradcam")
async def gradcam(file: UploadFile = File(...)):
    if model is None:
        # Demo mode: return a placeholder response so UI doesn't break
        return {
            "heatmap": None,
            "predicted_class": "N/A",
            "confidence": 0.0,
            "scores": {"Low": 0.0, "Medium": 0.0, "High": 0.0},
            "mock": True,
        }

    raw = await file.read()
    try:
        img = _load_image(raw)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not open uploaded image")

    result = _gradcam(img)
    return result
