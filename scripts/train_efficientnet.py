"""
Fine-tuned EfficientNet-B0 for thumbnail click-through prediction.

Uses timm to load a pretrained EfficientNet-B0, replaces the classifier head,
freezes early layers, and trains with early stopping.  Saves best weights and
training history.
"""

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "features.csv"
MODEL_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODEL_DIR / "efficientnet_best.pth"
HISTORY_PATH = MODEL_DIR / "training_history.json"

RANDOM_STATE = 42
BATCH_SIZE = 32
NUM_WORKERS = 0 if platform.system() == "Darwin" else 4
NUM_EPOCHS = 30
PATIENCE = 5
LR = 1e-4
NUM_CLASSES = 3
LABEL_MAP = {"Low": 0, "Medium": 1, "High": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ThumbnailDataset(Dataset):
    """Loads thumbnail images and returns (image_tensor, label)."""

    def __init__(self, dataframe: pd.DataFrame, transform: transforms.Compose):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row["thumbnail_path"]

        # Resolve relative paths against project root
        img_path = Path(img_path)
        if not img_path.is_absolute():
            img_path = PROJECT_ROOT / img_path

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        label = LABEL_MAP[row["label"]]
        return image, label


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1,
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
def build_model(device: torch.device) -> nn.Module:
    """Load EfficientNet-B0 and replace the classifier head."""
    model = timm.create_model("efficientnet_b0", pretrained=True)

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last 2 blocks (EfficientNet-B0 has blocks indexed 0-6)
    # timm stores blocks in model.blocks
    for block in model.blocks[-2:]:
        for param in block.parameters():
            param.requires_grad = True

    # Build new classifier head
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_CLASSES),
    )

    # Classifier head is trainable by default (new parameters)
    return model.to(device)


# ---------------------------------------------------------------------------
# Training loop helpers
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    total = len(all_labels)
    epoch_loss = running_loss / total
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # Device detection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded {len(df)} samples from {FEATURES_CSV}")

    # ------------------------------------------------------------------
    # 2. Stratified train / val / test  (70 / 15 / 15) — split by video
    # ------------------------------------------------------------------
    # Create a combined stratification key from label + niche
    df["_strat_key"] = df["label"] + "_" + df["niche"].astype(str)

    train_val_df, test_df = train_test_split(
        df,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=df["_strat_key"],
    )
    val_ratio = 0.15 / 0.85
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio,
        random_state=RANDOM_STATE,
        stratify=train_val_df["_strat_key"],
    )
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    # ------------------------------------------------------------------
    # 3. Data loaders
    # ------------------------------------------------------------------
    train_dataset = ThumbnailDataset(train_df, transform=train_transform)
    val_dataset = ThumbnailDataset(val_df, transform=eval_transform)
    test_dataset = ThumbnailDataset(test_df, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # ------------------------------------------------------------------
    # 4. Model, optimizer, scheduler, loss
    # ------------------------------------------------------------------
    model = build_model(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2,
    )
    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # 5. Training loop with early stopping
    # ------------------------------------------------------------------
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }
    best_val_loss = float("inf")
    epochs_no_improve = 0

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device,
        )
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        # Checkpoint best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    # Save training history
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {HISTORY_PATH}")

    # ------------------------------------------------------------------
    # 6. Final test evaluation using best weights
    # ------------------------------------------------------------------
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True))
    test_loss, test_acc, y_pred, y_true = evaluate(
        model, test_loader, criterion, device,
    )

    f1 = f1_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, average="macro")
    recall_val = recall_score(y_true, y_pred, average="macro")

    print("\n" + "=" * 50)
    print("EfficientNet-B0 – Test Metrics")
    print("=" * 50)
    print(f"  Accuracy:          {test_acc:.4f}")
    print(f"  F1 (macro):        {f1:.4f}")
    print(f"  Precision (macro): {precision:.4f}")
    print(f"  Recall (macro):    {recall_val:.4f}")
    print("=" * 50)
    print(f"\nBest model weights saved to {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
