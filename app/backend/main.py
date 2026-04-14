"""
ClickLens FastAPI Backend
Serves EfficientNet-B0 thumbnail CTR predictions and Grad-CAM explanations.
"""

import base64
import io
import logging
import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import anthropic as anthropic_sdk

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
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
MODEL_PATH        = Path(__file__).resolve().parent.parent.parent / "models" / "efficientnet_best.pth"
KNN_INDEX_PATH    = Path(__file__).resolve().parent.parent.parent / "models" / "knn_index.pkl"
EMBEDDINGS_MAP    = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "embeddings_mapping.csv"
RERANK_HEAD_PATH  = Path(__file__).resolve().parent.parent.parent / "models" / "rerank_head.pt"
TRAIN_EMBEDS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "embeddings.npy"
RERANK_HIDDEN     = 128
RERANK_CANDIDATES = 30

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
knn_index = None
embeddings_df: pd.DataFrame | None = None
rerank_head: nn.Module | None = None
train_embeds: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Rerank head — small MLP (1280 -> 128 -> 3) trained in scripts/train_rerank_head.py
# ---------------------------------------------------------------------------
class RerankHead(nn.Module):
    def __init__(self, in_dim: int = 1280, hidden: int = RERANK_HIDDEN, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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

    # Download model from URL if provided and file is missing
    model_url = os.getenv("MODEL_URL", "").strip()
    if not MODEL_PATH.exists() and model_url:
        logger.info("MODEL_URL set — downloading model from %s", model_url)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            import urllib.request
            urllib.request.urlretrieve(model_url, str(MODEL_PATH))
            logger.info("Model downloaded to %s (%d MB)",
                        MODEL_PATH, MODEL_PATH.stat().st_size // (1024 ** 2))
        except Exception as exc:
            logger.error("Failed to download model: %s", exc)

    if MODEL_PATH.exists():
        try:
            model = _build_model(device, MODEL_PATH)
            logger.info("Model loaded from %s", MODEL_PATH)
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            model = None
    else:
        logger.warning("Model file not found at %s – starting in demo mode", MODEL_PATH)
        model = None

    # Load kNN similarity index + mapping. build_index.py saves a dict artifact
    # {index, mapping, embed_dim}; we also tolerate an older bare-index format
    # that kept the mapping in a sibling CSV.
    global knn_index, embeddings_df
    if KNN_INDEX_PATH.exists():
        try:
            with open(KNN_INDEX_PATH, "rb") as f:
                artifact = pickle.load(f)

            if isinstance(artifact, dict) and "index" in artifact:
                knn_index = artifact["index"]
                embeddings_df = artifact.get("mapping")
                if embeddings_df is None and EMBEDDINGS_MAP.exists():
                    embeddings_df = pd.read_csv(EMBEDDINGS_MAP)
            else:
                # Legacy: pickle is the bare NearestNeighbors object
                knn_index = artifact
                embeddings_df = (
                    pd.read_csv(EMBEDDINGS_MAP) if EMBEDDINGS_MAP.exists() else None
                )

            if embeddings_df is None:
                raise RuntimeError("kNN mapping not available (no dict entry and no CSV)")
            logger.info("kNN index loaded (%d embeddings)", len(embeddings_df))
        except Exception as exc:
            logger.error("Failed to load kNN index: %s", exc)
            knn_index, embeddings_df = None, None
    else:
        logger.warning("kNN index not found – recommendations will run in demo mode")

    # ------------------------------------------------------------------
    # Optional MLP rerank head: reorders kNN candidates by predicted
    # CTR-class agreement with the query. Falls back silently to pure kNN
    # ordering when the checkpoint or training embeddings are missing.
    # ------------------------------------------------------------------
    global rerank_head, train_embeds

    rerank_url = os.getenv("RERANK_HEAD_URL", "").strip()
    if not RERANK_HEAD_PATH.exists() and rerank_url:
        logger.info("RERANK_HEAD_URL set — downloading rerank head from %s", rerank_url)
        RERANK_HEAD_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            import urllib.request
            urllib.request.urlretrieve(rerank_url, str(RERANK_HEAD_PATH))
        except Exception as exc:
            logger.error("Failed to download rerank head: %s", exc)

    if RERANK_HEAD_PATH.exists() and TRAIN_EMBEDS_PATH.exists():
        try:
            head = RerankHead()
            state = torch.load(str(RERANK_HEAD_PATH), map_location="cpu", weights_only=True)
            head.load_state_dict(state)
            head.eval()
            rerank_head = head
            train_embeds = np.load(TRAIN_EMBEDS_PATH).astype(np.float32)
            logger.info(
                "Rerank head loaded (%d training embeddings available)",
                len(train_embeds),
            )
        except Exception as exc:
            logger.error("Failed to load rerank head: %s", exc)
            rerank_head, train_embeds = None, None
    else:
        logger.warning(
            "Rerank head not found – /recommend will fall back to pure kNN ordering"
        )

    yield  # application is running


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ClickLens API",
    description="Thumbnail CTR prediction and Grad-CAM explanations",
    version="1.0.0",
    lifespan=lifespan,
)

_cors_raw = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000")
_cors_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
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


def _mock_recommend(niche: str) -> list:
    """Return plausible-looking mock recommendations for demo mode."""
    valid_niches = ["Gaming", "Travel", "Fitness"]
    n = niche if niche in valid_niches else "Gaming"
    return [
        {"niche": n, "CTR_label": "High", "similarity": round(0.95 - i * 0.06, 2), "video_id": f"demo_{i+1}"}
        for i in range(5)
    ]


def _mock_gradcam(img: Image.Image) -> str:
    """
    Generate a plausible-looking Grad-CAM overlay for demo mode.
    Uses a smooth random heatmap blended over the original image.
    Returns base64-encoded JPEG string.
    """
    img_np = np.array(img.convert("RGB"))
    h, w = img_np.shape[:2]

    # Build a smooth random attention map (low-res noise + upscale)
    rng_np = np.random.RandomState(42)
    low_res_h, low_res_w = max(h // 8, 4), max(w // 8, 4)
    noise = rng_np.rand(low_res_h, low_res_w).astype(np.float32)
    cam = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    cam = np.clip(cam, 0.0, 1.0)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = np.uint8(0.6 * img_np + 0.4 * heatmap_rgb)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", overlay_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


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


def _get_embedding(img: Image.Image) -> np.ndarray:
    """Extract the 1280-d EfficientNet backbone embedding for similarity search."""
    tensor = PREPROCESS(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.forward_features(tensor)   # (1, C, H, W)
        pooled   = model.global_pool(features)       # (1, 1280)
    return pooled.cpu().numpy().flatten()


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


@app.post("/embed")
async def embed(file: UploadFile = File(...)):
    """
    Return the 1280-d EfficientNet-B0 backbone embedding for a thumbnail.
    Useful as a standalone feature extractor for downstream retrieval or
    ranking models.
    """
    raw = await file.read()
    try:
        img = _load_image(raw)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not open uploaded image")

    if model is None:
        # Demo mode — return a deterministic pseudo-embedding so the endpoint
        # is still introspectable without a trained model.
        rng = np.random.RandomState(abs(hash(file.filename or "demo")) % (2**32))
        vec = rng.randn(1280).astype(np.float32)
        return {
            "embedding": vec.tolist(),
            "dim": 1280,
            "filename": file.filename,
            "mock": True,
        }

    embedding = _get_embedding(img)
    return {
        "embedding": embedding.astype(float).tolist(),
        "dim": int(embedding.shape[0]),
        "filename": file.filename,
        "mock": False,
    }


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
    raw = await file.read()
    try:
        img = _load_image(raw)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not open uploaded image")

    if model is None:
        # Demo mode: return a smooth random heatmap so the UI shows something useful
        b64 = _mock_gradcam(img)
        return {
            "heatmap": b64,
            "predicted_class": "N/A",
            "confidence": 0.0,
            "scores": {"Low": 0.0, "Medium": 0.0, "High": 0.0},
            "mock": True,
        }

    result = _gradcam(img)
    return result


@app.post("/recommend")
async def recommend(
    file: UploadFile = File(...),
    niche: str = Query(default=""),
):
    """
    Given a thumbnail and an optional niche, return the top-5 most similar
    High-CTR thumbnails from the training set (content-based recommendation).
    """
    raw = await file.read()
    try:
        img = _load_image(raw)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not open uploaded image")

    # Demo mode: no model or no index
    if model is None or knn_index is None or embeddings_df is None:
        return {"recommendations": _mock_recommend(niche), "mock": True}

    embedding = _get_embedding(img)
    n_neighbors = min(RERANK_CANDIDATES, len(embeddings_df))
    distances, indices = knn_index.kneighbors([embedding], n_neighbors=n_neighbors)

    cand_idx = indices[0]
    neighbours = embeddings_df.iloc[cand_idx].copy()
    neighbours = neighbours.assign(similarity=(1 - distances[0]).round(4))

    # ------------------------------------------------------------------
    # Optional rerank: score candidates by P(query's predicted CTR class)
    # using the MLP head. Fall back to pure similarity ordering otherwise.
    # ------------------------------------------------------------------
    reranked = False
    if rerank_head is not None and train_embeds is not None:
        try:
            with torch.no_grad():
                query_probs = F.softmax(
                    rerank_head(torch.from_numpy(embedding).float().unsqueeze(0)),
                    dim=1,
                ).cpu().numpy()[0]
                query_class = int(np.argmax(query_probs))

                cand_vecs = torch.from_numpy(train_embeds[cand_idx]).float()
                cand_probs = F.softmax(rerank_head(cand_vecs), dim=1).cpu().numpy()

            rerank_scores = cand_probs[:, query_class]
            neighbours = neighbours.assign(
                rerank_score=rerank_scores.round(4),
                query_predicted_class=CLASS_LABELS[query_class],
            )
            # Sort by rerank score desc; use similarity as a tiebreak
            neighbours = neighbours.sort_values(
                by=["rerank_score", "similarity"], ascending=[False, False]
            )
            reranked = True
        except Exception as exc:
            logger.error("Rerank failed, falling back to kNN order: %s", exc)

    # Filter by niche when provided
    if niche and niche.lower() != "other":
        by_niche = neighbours[neighbours["niche"].str.lower() == niche.lower()]
        neighbours = by_niche if len(by_niche) >= 2 else neighbours

    # Prefer High CTR; fall back to all if fewer than 2 High results
    high_ctr = neighbours[neighbours["CTR_label"] == "High"]
    top = high_ctr.head(5) if len(high_ctr) >= 2 else neighbours.head(5)

    recommendations = []
    for _, row in top.iterrows():
        rec = {
            "niche":      row["niche"],
            "CTR_label":  row["CTR_label"],
            "similarity": float(row["similarity"]),
            "video_id":   str(row.get("video_id", "")),
        }
        if reranked:
            rec["rerank_score"] = float(row["rerank_score"])
        recommendations.append(rec)

    return {
        "recommendations": recommendations,
        "mock": False,
        "reranked": reranked,
    }


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    niche: str = Query(default="Gaming"),
):
    """
    Send the thumbnail to Claude and return 3 bullet points of specific,
    actionable advice to improve click-through rate.
    Returns {"advice": null, "mock": true} when ANTHROPIC_API_KEY is not set.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return {"advice": None, "mock": True}

    raw = await file.read()
    try:
        img = _load_image(raw)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not open uploaded image")

    # Encode image as base64 JPEG for the Claude vision API
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    prompt = (
        f"You are a YouTube thumbnail expert. This thumbnail is from the {niche} niche. "
        "Give exactly 3 bullet points of specific, actionable advice to improve its "
        "click-through rate. Each bullet must reference something you can actually see "
        "(or is missing) in the image and suggest a concrete change. "
        "Topics to consider: text size and readability, color contrast, "
        "presence of a face or person, emotional appeal, and composition. "
        "Format: start each bullet with '•'. No intro sentence, no conclusion."
    )

    try:
        client = anthropic_sdk.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        advice = message.content[0].text
        return {"advice": advice, "mock": False}
    except Exception as exc:
        logger.error("Claude /analyze failed: %s", exc)
        return {"advice": None, "mock": True}
