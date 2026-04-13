"""
build_index.py — Build a cosine-similarity nearest-neighbor index on thumbnail embeddings.

Loads the EfficientNet-B0 embeddings produced by scripts/extract_embeddings.py,
fits a sklearn NearestNeighbors index under the cosine metric, and pickles the
fitted index together with its mapping so the serving backend can load a single
artifact at request time.

Outputs:
    models/knn_index.pkl  — dict with keys:
        index:     sklearn.neighbors.NearestNeighbors (fitted)
        mapping:   pandas.DataFrame (video_id, niche, CTR_label, thumbnail_path)
        embed_dim: int (1280)

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --n-neighbors 20
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings.npy"
MAPPING_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings_mapping.csv"
INDEX_PATH = PROJECT_ROOT / "models" / "knn_index.pkl"

DEFAULT_N_NEIGHBORS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a cosine-similarity kNN index on EfficientNet-B0 embeddings.",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=DEFAULT_N_NEIGHBORS,
        help="Default number of neighbors to return (default: %(default)s)",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=EMBEDDINGS_PATH,
        help="Path to embeddings .npy file",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=MAPPING_PATH,
        help="Path to mapping CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=INDEX_PATH,
        help="Path for the pickled index artifact",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.embeddings.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {args.embeddings}. "
            "Run scripts/extract_embeddings.py first."
        )
    if not args.mapping.exists():
        raise FileNotFoundError(
            f"Mapping not found at {args.mapping}. "
            "Run scripts/extract_embeddings.py first."
        )

    embeddings = np.load(args.embeddings).astype(np.float32)
    mapping = pd.read_csv(args.mapping)
    print(f"Loaded embeddings: shape={embeddings.shape}")
    print(f"Loaded mapping:    rows={len(mapping)}")

    if len(mapping) != embeddings.shape[0]:
        raise ValueError(
            f"Row mismatch: embeddings={embeddings.shape[0]} vs mapping={len(mapping)}."
        )

    n_neighbors = min(args.n_neighbors, len(embeddings))
    print(f"Fitting NearestNeighbors (cosine, n_neighbors={n_neighbors})...")
    index = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
    index.fit(embeddings)

    artifact = {
        "index": index,
        "mapping": mapping,
        "embed_dim": int(embeddings.shape[1]),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Security note: pickle artifacts must only be loaded from trusted,
    # access-controlled locations.  Loading a tampered file can execute
    # arbitrary code.  Ensure models/knn_index.pkl is never served from or
    # replaced via a publicly writable path.
    with open(args.output, "wb") as f:
        pickle.dump(artifact, f)
    print(f"Saved index artifact to {args.output}")


if __name__ == "__main__":
    main()
