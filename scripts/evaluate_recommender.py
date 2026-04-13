"""
evaluate_recommender.py — Retrieval-quality metrics for the EfficientNet-kNN
recommender.

For every thumbnail in the corpus we treat the thumbnail as a query, ask the
kNN index for its K nearest neighbours (excluding itself), and measure:

    Precision@K — fraction of retrieved items sharing the query's CTR_label.
    Recall@K    — retrieved relevant / total relevant (CTR_label matches
                  anywhere in the corpus, minus the query itself).

The numbers are directly comparable to the Random / NicheMeanCTR baselines
produced by scripts/train_baseline_recommender.py, so the output is a clean
"naive vs. deep-learning retriever" table for the rubric.

Outputs:
    data/outputs/knn_recommender_metrics.json

Usage:
    python scripts/evaluate_recommender.py
    python scripts/evaluate_recommender.py --ks 5 10 20
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings.npy"
INDEX_PATH = PROJECT_ROOT / "models" / "knn_index.pkl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"
METRICS_PATH = OUTPUT_DIR / "knn_recommender_metrics.json"
BASELINE_METRICS_PATH = OUTPUT_DIR / "baseline_recommender_metrics.json"

DEFAULT_KS = [5, 10, 20]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the EfficientNet-kNN recommender with P@K / R@K.",
    )
    parser.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS)
    parser.add_argument("--embeddings", type=Path, default=EMBEDDINGS_PATH)
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    parser.add_argument("--output", type=Path, default=METRICS_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.embeddings.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {args.embeddings}. "
            "Run scripts/extract_embeddings.py first."
        )
    if not args.index.exists():
        raise FileNotFoundError(
            f"kNN index not found at {args.index}. "
            "Run scripts/build_index.py first."
        )

    embeddings = np.load(args.embeddings).astype(np.float32)
    with open(args.index, "rb") as f:
        artifact = pickle.load(f)
    index = artifact["index"]
    mapping: pd.DataFrame = artifact["mapping"]

    if len(mapping) != embeddings.shape[0]:
        raise ValueError(
            f"Row mismatch: embeddings={embeddings.shape[0]} vs mapping={len(mapping)}."
        )

    labels = mapping["CTR_label"].to_numpy()
    label_counts = mapping["CTR_label"].value_counts().to_dict()
    n = len(mapping)
    max_k = max(args.ks)

    # Ask for k+1 because the nearest neighbor of every query is itself.
    print(f"Querying kNN index for top-{max_k + 1} neighbours of {n} items...")
    distances, indices = index.kneighbors(embeddings, n_neighbors=max_k + 1)

    # Drop the self-match: the first column is always the query itself
    # (distance 0). If the first column is not the query for some reason,
    # fall back to masking.
    trimmed_indices = np.empty((n, max_k), dtype=indices.dtype)
    for i in range(n):
        row = indices[i]
        row = row[row != i]            # mask any self-match
        trimmed_indices[i] = row[:max_k]

    retrieved_labels = labels[trimmed_indices]  # (n, max_k)

    results = {}
    for k in args.ks:
        top_k = retrieved_labels[:, :k]
        hits = (top_k == labels[:, None]).sum(axis=1)
        precision = hits / k
        # Total relevant = count of same-label items elsewhere in corpus
        total_relevant = np.array(
            [max(label_counts.get(lab, 0) - 1, 0) for lab in labels],
            dtype=np.float32,
        )
        # Avoid divide-by-zero (shouldn't happen in practice)
        recall = np.where(total_relevant > 0, hits / total_relevant, 0.0)
        results[k] = {
            "precision_at_k": float(precision.mean()),
            "recall_at_k": float(recall.mean()),
        }

    metrics = {
        "num_queries": n,
        "ks": args.ks,
        "model": "EfficientNet-B0 kNN (cosine)",
        "metrics": results,
        "relevance_rule": "retrieved item is relevant iff same CTR_label as query",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)

    # --- Pretty print ---
    print("\n" + "=" * 60)
    print("EfficientNet-kNN recommender metrics")
    print("=" * 60)
    for k in args.ks:
        m = results[k]
        print(
            f"  K={k:<3d}  P@K={m['precision_at_k']:.4f}  "
            f"R@K={m['recall_at_k']:.4f}"
        )
    print("=" * 60)
    print(f"\nMetrics saved to {args.output}")

    # --- Side-by-side vs baselines (if available) ---
    if BASELINE_METRICS_PATH.exists():
        with open(BASELINE_METRICS_PATH) as f:
            base = json.load(f)
        baselines = base.get("baselines", {})

        print("\n" + "=" * 60)
        print("Side-by-side: baselines vs EfficientNet-kNN (P@K / R@K)")
        print("=" * 60)
        header = f"{'model':<20s}" + "".join(f"  K={k:<3d}P/R      " for k in args.ks)
        print(header)
        for name, m in baselines.items():
            row = f"{name:<20s}"
            for k in args.ks:
                row += f"  {m[str(k)]['precision_at_k']:.3f}/{m[str(k)]['recall_at_k']:.3f}   "
            print(row)
        row = f"{'efficientnet_knn':<20s}"
        for k in args.ks:
            row += f"  {results[k]['precision_at_k']:.3f}/{results[k]['recall_at_k']:.3f}   "
        print(row)
        print("=" * 60)


if __name__ == "__main__":
    main()
