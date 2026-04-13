"""
train_baseline_recommender.py — Naive baselines for the thumbnail recommender.

Provides two non-learned baselines to benchmark against the EfficientNet-kNN
retrieval system:

    1. Random           — shuffle all other videos and take the top K.
    2. NicheMeanCTR     — for each query, return the top-K highest-CTR videos
                          within the same niche as the query (ties broken at
                          random). Approximates a "pick popular items per
                          category" strategy.

Relevance definition for retrieval metrics:
    A retrieved item is relevant to the query iff it shares the query's
    CTR_label. Precision@K and Recall@K are averaged over all query videos.

Outputs:
    data/outputs/baseline_recommender_metrics.json

Usage:
    python scripts/train_baseline_recommender.py
    python scripts/train_baseline_recommender.py --ks 5 10 20
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "features.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"
METRICS_PATH = OUTPUT_DIR / "baseline_recommender_metrics.json"

RANDOM_STATE = 42
DEFAULT_KS = [5, 10, 20]

# Numeric ordering used for the NicheMeanCTR baseline's sort
LABEL_ORDER = {"Low": 0, "Medium": 1, "High": 2}


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def precision_recall_at_k(
    retrieved_labels: np.ndarray,
    query_label: str,
    total_relevant: int,
    k: int,
) -> tuple[float, float]:
    """Precision@K and Recall@K for a single query."""
    effective_k = min(k, len(retrieved_labels))
    top_k = retrieved_labels[:effective_k]
    hits = int(np.sum(top_k == query_label))
    precision = hits / effective_k if effective_k > 0 else 0.0
    recall = hits / total_relevant if total_relevant > 0 else 0.0
    return precision, recall


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def rank_random(df: pd.DataFrame, query_idx: int, rng: np.random.Generator) -> np.ndarray:
    """Return a random permutation of all other row indices."""
    idx = np.arange(len(df))
    idx = idx[idx != query_idx]
    rng.shuffle(idx)
    return idx


def rank_niche_mean_ctr(
    df: pd.DataFrame,
    query_idx: int,
    niche_groups: dict[str, np.ndarray],
    label_scores: np.ndarray,
    rng: np.random.Generator,
    other_by_niche: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Rank items within the query's niche by label score (desc), then random.

    Items outside the niche are appended in random order so the ranking
    always has enough entries to evaluate at larger K values.

    Parameters
    ----------
    other_by_niche:
        Optional precomputed mapping from niche -> indices of all items *not*
        in that niche.  Pass this to avoid rebuilding the array on every call
        (see :func:`build_other_by_niche`).
    """
    query_niche = df.iloc[query_idx]["niche"]

    same_niche = niche_groups[query_niche]
    same_niche = same_niche[same_niche != query_idx]

    # Sort descending by label score with random tiebreaks
    tiebreak = rng.random(len(same_niche))
    order = np.lexsort((tiebreak, -label_scores[same_niche]))
    ranked_same = same_niche[order]

    if other_by_niche is not None:
        other = other_by_niche[query_niche].copy()
    else:
        other = np.concatenate(
            [idxs for niche, idxs in niche_groups.items() if niche != query_niche]
        ).astype(np.int64)
    rng.shuffle(other)

    return np.concatenate([ranked_same, other])


def build_other_by_niche(niche_groups: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Precompute, for each niche, the array of all indices *outside* that niche.

    Calling this once and passing the result to :func:`rank_niche_mean_ctr`
    avoids rebuilding the array on every query (O(N²) → O(N)).
    """
    return {
        niche: np.concatenate(
            [idxs for other_niche, idxs in niche_groups.items() if other_niche != niche]
        ).astype(np.int64, copy=False)
        for niche in niche_groups
    }


def evaluate_baseline(
    df: pd.DataFrame,
    ranker,
    ks: list[int],
    seed: int,
) -> dict[int, dict[str, float]]:
    """Run a ranker over every query video and average P@K / R@K."""
    rng = np.random.default_rng(seed)
    labels = df["CTR_label"].to_numpy()
    label_counts = df["CTR_label"].value_counts().to_dict()

    results = {k: {"precision": [], "recall": []} for k in ks}

    for q in range(len(df)):
        ranked_idx = ranker(q, rng)
        retrieved_labels = labels[ranked_idx]
        query_label = labels[q]
        # Total relevant in the corpus excluding the query itself
        total_relevant = label_counts.get(query_label, 0) - 1

        for k in ks:
            p, r = precision_recall_at_k(retrieved_labels, query_label, total_relevant, k)
            results[k]["precision"].append(p)
            results[k]["recall"].append(r)

    return {
        k: {
            "precision_at_k": float(np.mean(results[k]["precision"])),
            "recall_at_k": float(np.mean(results[k]["recall"])),
        }
        for k in ks
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Naive baseline recommenders (random + niche-mean-CTR).",
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=DEFAULT_KS,
        help="K values for precision@K / recall@K (default: %(default)s)",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=FEATURES_CSV,
        help="Path to features.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=METRICS_PATH,
        help="Where to save the metrics JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.features)
    print(f"Loaded {len(df)} rows from {args.features}")
    required = {"niche", "CTR_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"features.csv is missing required columns: {sorted(missing)}. "
            "Re-run scripts/build_features.py to refresh the schema."
        )

    # Precompute per-niche index groups and a numeric label score for fast ranking
    niche_groups = {
        niche: df.index[df["niche"] == niche].to_numpy()
        for niche in df["niche"].unique()
    }
    label_scores = df["CTR_label"].map(LABEL_ORDER).fillna(0).to_numpy()
    other_by_niche = build_other_by_niche(niche_groups)

    # --- Random baseline ---
    print("Evaluating Random baseline...")
    random_metrics = evaluate_baseline(
        df,
        ranker=lambda q, rng: rank_random(df, q, rng),
        ks=args.ks,
        seed=args.seed,
    )

    # --- Niche-mean-CTR baseline ---
    print("Evaluating NicheMeanCTR baseline...")
    niche_metrics = evaluate_baseline(
        df,
        ranker=lambda q, rng: rank_niche_mean_ctr(
            df, q, niche_groups, label_scores, rng, other_by_niche=other_by_niche,
        ),
        ks=args.ks,
        seed=args.seed,
    )

    metrics = {
        "num_queries": len(df),
        "ks": args.ks,
        "baselines": {
            "random": random_metrics,
            "niche_mean_ctr": niche_metrics,
        },
        "relevance_rule": "retrieved item is relevant iff same CTR_label as query",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)

    # --- Pretty print ---
    print("\n" + "=" * 60)
    print("Baseline recommender metrics")
    print("=" * 60)
    for name, m in metrics["baselines"].items():
        print(f"\n[{name}]")
        for k in args.ks:
            print(
                f"  K={k:<3d}  P@K={m[k]['precision_at_k']:.4f}  "
                f"R@K={m[k]['recall_at_k']:.4f}"
            )
    print("=" * 60)
    print(f"\nMetrics saved to {args.output}")


if __name__ == "__main__":
    main()
