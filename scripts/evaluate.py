"""
Unified evaluation of all three ClickLens models.

Loads the baseline, XGBoost, and EfficientNet models and evaluates them on
the **same** held-out test split (random_state=42).  Generates confusion
matrices and a comparison CSV.
"""

import pickle
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "features.csv"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"

BASELINE_PATH = MODEL_DIR / "baseline_model.pkl"
XGBOOST_PATH = MODEL_DIR / "xgboost_model.pkl"
LABEL_ENC_PATH = MODEL_DIR / "label_encoder.pkl"
EFFNET_PATH = MODEL_DIR / "efficientnet_best.pth"

RANDOM_STATE = 42
BATCH_SIZE = 32
NUM_WORKERS = 0 if platform.system() == "Darwin" else 4
NUM_CLASSES = 3
LABEL_MAP = {"Low": 0, "Medium": 1, "High": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
CLASS_NAMES = ["Low", "Medium", "High"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

FEATURE_COLS = [
    "dominant_color_1_r", "dominant_color_1_g", "dominant_color_1_b",
    "dominant_color_2_r", "dominant_color_2_g", "dominant_color_2_b",
    "dominant_color_3_r", "dominant_color_3_g", "dominant_color_3_b",
    "brightness", "contrast", "saturation",
    "face_count", "face_area_ratio",
    "text_present", "text_area_ratio",
    "edge_density", "color_entropy",
]


# ---------------------------------------------------------------------------
# EfficientNet helpers (mirror train_efficientnet.py)
# ---------------------------------------------------------------------------
class ThumbnailDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(row["thumbnail_path"])
        if not img_path.is_absolute():
            img_path = PROJECT_ROOT / img_path
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = LABEL_MAP[row["label"]]
        return image, label


eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def build_efficientnet(device: torch.device) -> nn.Module:
    model = timm.create_model("efficientnet_b0", pretrained=False)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_CLASSES),
    )
    return model.to(device)


@torch.no_grad()
def efficientnet_predict(
    model: nn.Module, loader: DataLoader, device: torch.device,
):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


# ---------------------------------------------------------------------------
# Confusion matrix plotting
# ---------------------------------------------------------------------------
def save_confusion_matrix(
    y_true, y_pred, class_names, title: str, save_path: Path,
):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved to {save_path}")


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data & create the SAME splits used during training
    # ------------------------------------------------------------------
    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded {len(df)} samples\n")

    # Label-encode for XGBoost / EfficientNet evaluation
    le = LabelEncoder()
    le.classes_ = np.array(["Low", "Medium", "High"])
    df["label_enc"] = le.transform(df["label"])

    # Combined stratification key (label + niche) — matches train_efficientnet
    df["_strat_key"] = df["label"] + "_" + df["niche"].astype(str)

    # 70 / 15 / 15 split (same random_state everywhere)
    train_val_df, test_df = train_test_split(
        df, test_size=0.15, random_state=RANDOM_STATE, stratify=df["_strat_key"],
    )
    val_ratio = 0.15 / 0.85
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_ratio, random_state=RANDOM_STATE,
        stratify=train_val_df["_strat_key"],
    )
    print(f"Test set size: {len(test_df)}")

    # Also create the 70/30 baseline split
    train_bl, test_bl = train_test_split(
        df, test_size=0.30, random_state=RANDOM_STATE, stratify=df["label"],
    )

    results = {}

    # ------------------------------------------------------------------
    # 2. Baseline
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BASELINE (Majority Class)")
    print("=" * 60)
    with open(BASELINE_PATH, "rb") as f:
        baseline = pickle.load(f)
    majority_class = baseline["majority_class"]
    y_true_bl = test_bl["label"].values
    y_pred_bl = [majority_class] * len(y_true_bl)
    metrics_bl = compute_metrics(y_true_bl, y_pred_bl)
    results["Baseline"] = metrics_bl
    for k, v in metrics_bl.items():
        print(f"  {k:20s}: {v:.4f}")

    # ------------------------------------------------------------------
    # 3. XGBoost
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("XGBOOST")
    print("=" * 60)
    with open(XGBOOST_PATH, "rb") as f:
        xgb_model = pickle.load(f)

    X_test_xgb = test_df[FEATURE_COLS].values
    y_test_xgb = test_df["label_enc"].values
    y_pred_xgb = xgb_model.predict(X_test_xgb)

    metrics_xgb = compute_metrics(y_test_xgb, y_pred_xgb)
    results["XGBoost"] = metrics_xgb
    for k, v in metrics_xgb.items():
        print(f"  {k:20s}: {v:.4f}")

    save_confusion_matrix(
        y_test_xgb, y_pred_xgb, CLASS_NAMES,
        "XGBoost Confusion Matrix",
        OUTPUT_DIR / "confusion_matrix_xgboost.png",
    )

    # ------------------------------------------------------------------
    # 4. EfficientNet
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EFFICIENTNET-B0")
    print("=" * 60)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"  Device: {device}")

    eff_model = build_efficientnet(device)
    eff_model.load_state_dict(
        torch.load(EFFNET_PATH, map_location=device, weights_only=True),
    )

    test_dataset = ThumbnailDataset(test_df, transform=eval_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    y_pred_eff, y_true_eff = efficientnet_predict(eff_model, test_loader, device)

    metrics_eff = compute_metrics(y_true_eff, y_pred_eff)
    results["EfficientNet"] = metrics_eff
    for k, v in metrics_eff.items():
        print(f"  {k:20s}: {v:.4f}")

    save_confusion_matrix(
        y_true_eff, y_pred_eff, CLASS_NAMES,
        "EfficientNet-B0 Confusion Matrix",
        OUTPUT_DIR / "confusion_matrix_efficientnet.png",
    )

    # ------------------------------------------------------------------
    # 5. Comparison table
    # ------------------------------------------------------------------
    comparison_df = pd.DataFrame(results).T
    comparison_df.index.name = "model"
    comparison_df = comparison_df.reset_index()

    csv_path = OUTPUT_DIR / "model_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"\nComparison table saved to {csv_path}\n")

    # Pretty-print
    print("=" * 72)
    print(f"{'Model':<18} {'Accuracy':>10} {'F1 Macro':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 72)
    for _, row in comparison_df.iterrows():
        print(
            f"{row['model']:<18} "
            f"{row['accuracy']:>10.4f} "
            f"{row['f1_macro']:>10.4f} "
            f"{row['precision_macro']:>10.4f} "
            f"{row['recall_macro']:>10.4f}"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
