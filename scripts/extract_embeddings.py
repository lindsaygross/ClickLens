"""
extract_embeddings.py — Dump EfficientNet-B0 backbone embeddings for every thumbnail.

Loads the fine-tuned EfficientNet-B0 from models/efficientnet_best.pth, replaces
the classification head with Identity so the forward pass returns the 1280-d
pooled backbone features, and runs every thumbnail in features.csv through it.

Outputs:
    data/processed/embeddings.npy       (N, 1280) float32
    data/processed/embeddings_mapping.csv
        columns: video_id, niche, CTR_label, thumbnail_path

Usage:
    python scripts/extract_embeddings.py
"""

import platform
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "features.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "efficientnet_best.pth"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings.npy"
MAPPING_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings_mapping.csv"

BATCH_SIZE = 32
NUM_WORKERS = 0 if platform.system() == "Darwin" else 4
EMBED_DIM = 1280

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ThumbnailOnlyDataset(Dataset):
    """Yields (image_tensor, row_index) so we can keep embeddings aligned to df."""

    def __init__(self, dataframe: pd.DataFrame, transform: transforms.Compose):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        img_path = Path(self.df.iloc[idx]["thumbnail_path"])
        if not img_path.is_absolute():
            img_path = PROJECT_ROOT / img_path
        image = Image.open(img_path).convert("RGB")
        return self.transform(image), idx


eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
def load_backbone(device: torch.device) -> nn.Module:
    """Rebuild train_efficientnet's architecture, load weights, strip the head.

    Falls back to ImageNet-pretrained backbone weights if the fine-tuned
    checkpoint is missing — the classifier head is stripped either way, so
    retrieval-quality embeddings are still meaningful.
    """
    if MODEL_PATH.exists():
        print(f"Loading fine-tuned weights from {MODEL_PATH}")
        model = timm.create_model("efficientnet_b0", pretrained=False)
        in_features = model.classifier.in_features  # 1280

        # Match training-time head so the checkpoint loads cleanly
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3),
        )
        state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        model.load_state_dict(state)
    else:
        print(
            f"[warn] No fine-tuned checkpoint at {MODEL_PATH}; "
            "falling back to ImageNet-pretrained EfficientNet-B0 backbone."
        )
        model = timm.create_model("efficientnet_b0", pretrained=True)

    # Replace head with Identity so forward() returns 1280-d pooled features
    model.classifier = nn.Identity()
    model.eval()
    return model.to(device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@torch.no_grad()
def main() -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded {len(df)} rows from {FEATURES_CSV}")

    required = {"video_id", "niche", "CTR_label", "thumbnail_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"features.csv is missing required columns: {sorted(missing)}. "
            "Re-run scripts/build_features.py to refresh the schema."
        )

    model = load_backbone(device)

    dataset = ThumbnailOnlyDataset(df, transform=eval_transform)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    embeddings = np.zeros((len(dataset), EMBED_DIM), dtype=np.float32)

    for images, indices in tqdm(loader, desc="Extracting embeddings"):
        images = images.to(device)
        feats = model(images).cpu().numpy().astype(np.float32)
        for j, idx in enumerate(indices.tolist()):
            embeddings[idx] = feats[j]

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"Saved embeddings: {EMBEDDINGS_PATH}  shape={embeddings.shape}")

    mapping = df[["video_id", "niche", "CTR_label", "thumbnail_path"]].copy()
    mapping.to_csv(MAPPING_PATH, index=False)
    print(f"Saved mapping:    {MAPPING_PATH}  rows={len(mapping)}")


if __name__ == "__main__":
    main()
