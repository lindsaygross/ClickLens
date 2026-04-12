"""
build_features.py — Feature engineering and labeling pipeline for ClickLens.

Reads video_metadata.csv and thumbnail images, extracts visual features from
each thumbnail, computes a performance-based label, and writes the combined
result to data/processed/features.csv.

Usage:
    python scripts/build_features.py
    python scripts/build_features.py --input data/processed/video_metadata.csv \
                                     --output data/processed/features.csv
"""

import argparse
import logging
import os
import warnings
from pathlib import Path

# Prevent segfault from OMP/MKL threading conflict between PyTorch and scikit-learn on macOS
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import easyocr
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.cluster import KMeans
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress noisy third-party warnings (e.g. easyocr / torch)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "video_metadata.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "features.csv"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _load_image(path: Path):
    """Load an image via OpenCV.  Returns None on failure."""
    img = cv2.imread(str(path))
    if img is None:
        logger.warning("Could not load image: %s", path)
    return img


def extract_dominant_colors(img: np.ndarray, k: int = 3) -> list[float]:
    """Return the k dominant RGB colors via K-means on pixel values.

    Returns a flat list of length k*3:
        [r1, g1, b1, r2, g2, b2, r3, g3, b3]
    """
    # Reshape to (num_pixels, 3) and convert BGR -> RGB
    pixels = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(np.float32)

    # Sub-sample for speed when the image is large
    max_pixels = 10_000
    if len(pixels) > max_pixels:
        indices = np.random.choice(len(pixels), max_pixels, replace=False)
        pixels = pixels[indices]

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(pixels)

    # Sort clusters by frequency (most dominant first)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    order = np.argsort(-counts)
    centers = kmeans.cluster_centers_[order]

    return centers.flatten().tolist()


def extract_brightness(gray: np.ndarray) -> float:
    """Mean of grayscale pixel values (0-255 scale)."""
    return float(np.mean(gray))


def extract_contrast(gray: np.ndarray) -> float:
    """Standard deviation of grayscale pixel values."""
    return float(np.std(gray))


def extract_saturation(img: np.ndarray) -> float:
    """Mean saturation in HSV color space."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1]))


def extract_face_features(
    gray: np.ndarray, face_cascade: cv2.CascadeClassifier
) -> tuple[int, float]:
    """Detect faces and return (face_count, face_area_ratio)."""
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    face_count = len(faces)
    if face_count == 0:
        return 0, 0.0

    total_face_area = sum(int(w) * int(h) for (_, _, w, h) in faces)
    image_area = gray.shape[0] * gray.shape[1]
    face_area_ratio = total_face_area / image_area
    return face_count, float(face_area_ratio)


def extract_text_features(
    img: np.ndarray, reader: easyocr.Reader
) -> tuple[int, float]:
    """Detect text via EasyOCR and return (text_present, text_area_ratio).

    text_present: 1 if any text detected, 0 otherwise.
    text_area_ratio: sum of text bounding-box areas / image area.
    """
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = reader.readtext(rgb)

    if not results:
        return 0, 0.0

    image_area = img.shape[0] * img.shape[1]
    total_text_area = 0.0
    for bbox, _text, _conf in results:
        # bbox is a list of 4 corner points [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        total_text_area += w * h

    text_area_ratio = total_text_area / image_area
    return 1, float(text_area_ratio)


def extract_edge_density(gray: np.ndarray) -> float:
    """Proportion of edge pixels (Canny) to total pixels."""
    edges = cv2.Canny(gray, 100, 200)
    return float(np.count_nonzero(edges) / edges.size)


def extract_color_entropy(img: np.ndarray) -> float:
    """Shannon entropy of the flattened color histogram."""
    # Compute a 3-D histogram in RGB space and flatten
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256]
    )
    hist = hist.flatten()
    # Normalize to probability distribution
    hist = hist / (hist.sum() + 1e-12)
    return float(entropy(hist))


# ---------------------------------------------------------------------------
# Main feature extraction for one image
# ---------------------------------------------------------------------------

def extract_all_features(
    img: np.ndarray,
    face_cascade: cv2.CascadeClassifier,
    ocr_reader: easyocr.Reader,
) -> dict:
    """Extract all thumbnail features from a single image.

    Returns a dict with all feature column names -> values.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Dominant colors
    dom_colors = extract_dominant_colors(img, k=3)
    color_keys = [
        f"dominant_color_{i}_{c}"
        for i in range(1, 4)
        for c in ("r", "g", "b")
    ]
    features = dict(zip(color_keys, dom_colors))

    # 2. Brightness
    features["brightness"] = extract_brightness(gray)

    # 3. Contrast
    features["contrast"] = extract_contrast(gray)

    # 4. Saturation
    features["saturation"] = extract_saturation(img)

    # 5-6. Face count & face area ratio
    face_count, face_area_ratio = extract_face_features(gray, face_cascade)
    features["face_count"] = face_count
    features["face_area_ratio"] = face_area_ratio

    # 7-8. Text present & text area ratio
    text_present, text_area_ratio = extract_text_features(img, ocr_reader)
    features["text_present"] = text_present
    features["text_area_ratio"] = text_area_ratio

    # 9. Edge density
    features["edge_density"] = extract_edge_density(gray)

    # 10. Color entropy
    features["color_entropy"] = extract_color_entropy(img)

    return features


# ---------------------------------------------------------------------------
# Labeling helpers
# ---------------------------------------------------------------------------

def compute_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'label' column based on performance terciles.

    Steps:
        1. performance_ratio = view_count / subscriber_count
        2. normalized_performance = performance_ratio / log(video_age_days + 1)
        3. Bucket into terciles: Low / Medium / High
    """
    df = df.copy()

    # Handle division by zero for subscriber_count
    df["performance_ratio"] = np.where(
        df["subscriber_count"] > 0,
        df["view_count"] / df["subscriber_count"],
        0.0,
    )

    # Normalize by video age
    df["normalized_performance"] = df["performance_ratio"] / np.log(
        df["video_age_days"].astype(float) + 1
    )

    # Handle any NaN / inf values that slipped through
    df["normalized_performance"] = df["normalized_performance"].replace(
        [np.inf, -np.inf], np.nan
    )
    df["normalized_performance"] = df["normalized_performance"].fillna(0.0)

    # Tercile bucketing
    tercile_labels = ["Low", "Medium", "High"]
    df["label"] = pd.qcut(
        df["normalized_performance"],
        q=3,
        labels=tercile_labels,
        duplicates="drop",
    )

    # Drop intermediate columns (keep the final label only)
    df.drop(columns=["performance_ratio", "normalized_performance"], inplace=True)

    return df


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_features(input_csv: Path, output_csv: Path) -> None:
    """End-to-end feature engineering + labeling pipeline."""

    # ------------------------------------------------------------------
    # 1. Load metadata
    # ------------------------------------------------------------------
    logger.info("Loading metadata from %s", input_csv)
    meta_df = pd.read_csv(input_csv)
    logger.info("Loaded %d rows", len(meta_df))

    # ------------------------------------------------------------------
    # 2. Initialise heavy resources once
    # ------------------------------------------------------------------
    logger.info("Initialising OpenCV Haar cascade for face detection...")
    haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(haar_path)
    if face_cascade.empty():
        raise RuntimeError(
            f"Failed to load Haar cascade from {haar_path}. "
            "Ensure OpenCV is installed with data files."
        )

    logger.info("Initialising EasyOCR reader (this may take a moment)...")
    ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    # ------------------------------------------------------------------
    # 3. Extract features for every thumbnail
    # ------------------------------------------------------------------
    feature_records: list[dict] = []
    skipped = 0

    for idx, row in tqdm(
        meta_df.iterrows(), total=len(meta_df), desc="Extracting features"
    ):
        video_id = row["video_id"]
        niche = row["niche"]

        # Resolve thumbnail path
        thumb_path = RAW_DATA_DIR / str(niche) / f"{video_id}.jpg"

        # Also try the path stored in metadata if the canonical one is missing
        if not thumb_path.exists() and pd.notna(row.get("thumbnail_path")):
            alt = Path(row["thumbnail_path"])
            if alt.is_absolute() and alt.exists():
                thumb_path = alt
            else:
                # Try relative to project root
                candidate = PROJECT_ROOT / alt
                if candidate.exists():
                    thumb_path = candidate

        if not thumb_path.exists():
            logger.warning(
                "Thumbnail not found for video %s (tried %s) — skipping",
                video_id,
                thumb_path,
            )
            skipped += 1
            continue

        img = _load_image(thumb_path)
        if img is None:
            skipped += 1
            continue

        try:
            feats = extract_all_features(img, face_cascade, ocr_reader)
        except Exception:
            logger.warning(
                "Feature extraction failed for video %s — skipping",
                video_id,
                exc_info=True,
            )
            skipped += 1
            continue

        # Carry over all original metadata columns
        record = row.to_dict()
        record.update(feats)
        feature_records.append(record)

    logger.info(
        "Feature extraction complete: %d succeeded, %d skipped",
        len(feature_records),
        skipped,
    )

    if not feature_records:
        logger.error("No features extracted — nothing to write. Exiting.")
        return

    features_df = pd.DataFrame(feature_records)

    # ------------------------------------------------------------------
    # 4. Labeling
    # ------------------------------------------------------------------
    logger.info("Computing performance labels...")
    features_df = compute_labels(features_df)

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_csv, index=False)
    logger.info("Saved features CSV to %s (%d rows)", output_csv, len(features_df))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ClickLens — thumbnail feature engineering & labeling pipeline",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the video metadata CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the output features CSV (default: %(default)s)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_features(input_csv=args.input, output_csv=args.output)
