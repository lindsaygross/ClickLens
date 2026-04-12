"""
Grad-CAM visualization for EfficientNet-B0 thumbnail classifier.

Produces heatmap overlays showing which regions of a YouTube thumbnail
drove the model's prediction. Can be used from the CLI or imported by
the backend API via `generate_gradcam()`.
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "efficientnet_best.pth"

CLASS_NAMES = ["Low", "Medium", "High"]

# ---------------------------------------------------------------------------
# Preprocessing (must match train_efficientnet.py)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model builder (mirrors train_efficientnet.py)
# ---------------------------------------------------------------------------
def build_model(device: torch.device) -> nn.Module:
    """Build EfficientNet-B0 with the custom classifier head used during
    training and load the saved weights."""
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
    in_features = model.num_features  # 1280 for efficientnet_b0
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 3),
    )
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
class GradCAM:
    """Grad-CAM implementation targeting a specific convolutional layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Register hooks
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(
            self._save_gradient
        )

    # -- hook callbacks -----------------------------------------------------
    def _save_activation(
        self,
        module: nn.Module,
        input: Tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self.activations = output.detach()

    def _save_gradient(
        self,
        module: nn.Module,
        grad_input: Tuple[torch.Tensor, ...],
        grad_output: Tuple[torch.Tensor, ...],
    ) -> None:
        self.gradients = grad_output[0].detach()

    # -- main entry ---------------------------------------------------------
    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """Run Grad-CAM.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Pre-processed image batch of shape (1, 3, 224, 224).
        class_idx : int or None
            Class to visualize.  If None, the predicted class is used.

        Returns
        -------
        heatmap : np.ndarray  (H_feat, W_feat) float32 in [0, 1]
        predicted_class : int
        probabilities : np.ndarray  (num_classes,) float32
        """
        self.model.zero_grad()
        output = self.model(input_tensor)  # (1, num_classes)
        probabilities = torch.softmax(output, dim=1).detach().cpu().numpy().squeeze()

        predicted_class = int(output.argmax(dim=1).item())
        target_class = class_idx if class_idx is not None else predicted_class

        # Back-propagate w.r.t. the target class score
        score = output[0, target_class]
        score.backward()

        # Compute Grad-CAM weights: global-average-pool the gradients
        gradients = self.gradients  # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination followed by ReLU
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()  # (H, W)

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, predicted_class, probabilities

    def remove_hooks(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()


# ---------------------------------------------------------------------------
# Public API — used by the backend
# ---------------------------------------------------------------------------
def generate_gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    original_image: np.ndarray,
    class_idx: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Produce a Grad-CAM heatmap and overlay for a single image.

    Parameters
    ----------
    model : nn.Module
        Loaded EfficientNet-B0 model (eval mode, on correct device).
    image_tensor : torch.Tensor
        Pre-processed tensor of shape (1, 3, 224, 224) on the model's device.
    original_image : np.ndarray
        Original image as (H, W, 3) uint8 BGR or RGB.
    class_idx : int or None
        Which class to visualize.  None = predicted class.

    Returns
    -------
    heatmap : np.ndarray (H, W, 3) uint8  — colorized heatmap at original size
    overlay : np.ndarray (H, W, 3) uint8  — blended heatmap + original
    predicted_class : int
    probabilities : np.ndarray (num_classes,) float32
    """
    # Target layer: conv_head is the last 1x1 conv before global avg pool
    target_layer = model.conv_head
    gradcam = GradCAM(model, target_layer)

    try:
        cam, predicted_class, probabilities = gradcam(image_tensor, class_idx)
    finally:
        gradcam.remove_hooks()

    h, w = original_image.shape[:2]

    # Resize CAM to original image size
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    cam_uint8 = np.uint8(255 * cam_resized)

    # Apply JET colormap
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)  # BGR

    # Build overlay (blend heatmap with original image)
    # Ensure original_image is BGR for consistent blending with cv2 colormap
    if original_image.shape[2] == 3:
        orig_bgr = original_image.copy()
    else:
        orig_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGBA2BGR)

    overlay = cv2.addWeighted(orig_bgr, 0.55, heatmap, 0.45, 0)

    return heatmap, overlay, predicted_class, probabilities


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM heatmap for a YouTube thumbnail."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input thumbnail image.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gradcam_output.jpg",
        help="Path to save the heatmap overlay (default: gradcam_output.jpg).",
    )
    parser.add_argument(
        "--class-idx",
        type=int,
        default=None,
        help="Class index to visualize (0=Low, 1=Medium, 2=High). "
        "Default: predicted class.",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = get_device()
    print(f"Device: {device}")

    # Load model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found at {MODEL_PATH}. "
            "Run train_efficientnet.py first."
        )
    model = build_model(device)
    print(f"Model loaded from {MODEL_PATH}")

    # Load and preprocess image
    pil_image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(pil_image).unsqueeze(0).to(device)

    # Keep original image as numpy BGR for overlay
    original_np = cv2.imread(str(image_path))
    if original_np is None:
        raise ValueError(f"cv2 could not read image: {image_path}")

    # Generate Grad-CAM
    heatmap, overlay, pred_class, probs = generate_gradcam(
        model, input_tensor, original_np, class_idx=args.class_idx
    )

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)

    # Print results
    target_label = args.class_idx if args.class_idx is not None else pred_class
    print(f"\nPredicted class: {CLASS_NAMES[pred_class]} (index {pred_class})")
    print(f"Visualizing class: {CLASS_NAMES[target_label]} (index {target_label})")
    print("Class probabilities:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {probs[i]:.4f}")
    print(f"\nHeatmap overlay saved to {output_path}")


if __name__ == "__main__":
    main()
