"""
create_placeholder_model.py
----------------------------
Generates a placeholder efficientnet_best.pth with the correct architecture
and random weights. Use this to enable deployment before the real trained
model is available. Predictions will be random but the app runs fully.

Run from the repo root:
    python scripts/create_placeholder_model.py
"""

from pathlib import Path

import torch
import torch.nn as nn
import timm


def build_model() -> nn.Module:
    net = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
    num_features = net.num_features  # 1280
    net.classifier = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 3),
    )
    return net


def main():
    out_path = Path(__file__).resolve().parent.parent / "models" / "efficientnet_best.pth"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"[skip] {out_path} already exists — not overwriting.")
        print("       Delete it manually if you want to regenerate.")
        return

    print("Building EfficientNet-B0 with random weights ...")
    model = build_model()
    torch.save(model.state_dict(), str(out_path))
    size_mb = out_path.stat().st_size / (1024 ** 2)
    print(f"[ok]  Saved placeholder to {out_path}  ({size_mb:.1f} MB)")
    print()
    print("NOTE: This model has RANDOM weights — predictions are meaningless.")
    print("      Replace this file with the real trained model before submission.")


if __name__ == "__main__":
    main()
