"""
train_rerank_head.py — Small MLP ranking head on top of EfficientNet embeddings.

Builds a two-stage recommender:
    1. Retrieval   — kNN over EfficientNet-B0 embeddings (cosine).
    2. Reranking   — an MLP (1280 -> 128 -> 3) predicts the CTR class, and
                     candidates are reordered by the probability of the
                     *query's* predicted class (class-agreement rerank).

Training signal:
    Cross-entropy on the 3-class CTR_label (Low / Medium / High).  The head is
    trained on precomputed embeddings only, so the whole script runs on CPU in
    seconds without touching image pixels.

Evaluation:
    For every query in the test split, run kNN to fetch the top-30 candidates
    from the *training* embeddings, then rerank them by the head's score.
    Report P@K / R@K with the same relevance rule used for the baselines
    (retrieved item is relevant iff same CTR_label as the query).  Output is
    directly comparable to baseline_recommender_metrics.json and
    knn_recommender_metrics.json.

Outputs:
    models/rerank_head.pt
    data/outputs/rerank_recommender_metrics.json

Usage:
    python scripts/train_rerank_head.py
    python scripts/train_rerank_head.py --epochs 30 --hidden 256
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

LABEL_MAP = {"Low": 0, "Medium": 1, "High": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES = 3

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings.npy"
MAPPING_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings_mapping.csv"
HEAD_PATH = PROJECT_ROOT / "models" / "rerank_head.pt"
METRICS_PATH = PROJECT_ROOT / "data" / "outputs" / "rerank_recommender_metrics.json"

RANDOM_STATE = 42
DEFAULT_KS = [5, 10, 20]
DEFAULT_CANDIDATES = 30   # retrieval depth before reranking


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class RerankHead(nn.Module):
    """Tiny MLP: embedding -> 3-class CTR logits."""

    def __init__(self, in_dim: int = 1280, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def precision_recall_at_k(
    retrieved_labels: np.ndarray,
    query_label: str,
    total_relevant: int,
    k: int,
) -> tuple[float, float]:
    effective_k = min(k, len(retrieved_labels))
    top_k = retrieved_labels[:effective_k]
    hits = int(np.sum(top_k == query_label))
    precision = hits / effective_k if effective_k > 0 else 0.0
    recall = hits / total_relevant if total_relevant > 0 else 0.0
    return precision, recall


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_head(
    head: RerankHead,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
) -> dict:
    """Train with CrossEntropy, early-stop on val loss, return history."""
    device = torch.device("cpu")  # tiny model, CPU is faster than MPS here
    head = head.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_val = float("inf")
    patience_left = 5

    n = X_train.shape[0]
    for epoch in range(1, epochs + 1):
        head.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb = X_train[idx].to(device)
            yb = y_train[idx].to(device)

            optimizer.zero_grad()
            logits = head(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        train_loss = total_loss / n

        head.eval()
        with torch.no_grad():
            val_logits = head(X_val.to(device))
            val_loss = criterion(val_logits, y_val.to(device)).item()
            val_preds = val_logits.argmax(dim=1).cpu().numpy()
            val_true = y_val.cpu().numpy()
            val_acc = accuracy_score(val_true, val_preds)
            val_f1 = f1_score(val_true, val_preds, average="macro", zero_division=0)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(
            f"  epoch {epoch:02d}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}"
        )

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            patience_left = 5
            torch.save(head.state_dict(), HEAD_PATH)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  early stopping at epoch {epoch}")
                break

    # Load best
    head.load_state_dict(torch.load(HEAD_PATH, map_location=device, weights_only=True))
    return history


# ---------------------------------------------------------------------------
# Retrieval + rerank evaluation
# ---------------------------------------------------------------------------
def evaluate_rerank(
    head: RerankHead,
    train_embeds: np.ndarray,
    train_labels: np.ndarray,
    test_embeds: np.ndarray,
    test_labels: np.ndarray,
    ks: list[int],
    n_candidates: int,
) -> tuple[dict, dict]:
    """For each test query, retrieve n_candidates from train via kNN, then rerank.

    Returns (kNN-only metrics, reranked metrics) so the lift is easy to report.
    """
    print(f"Fitting training kNN index ({len(train_embeds)} items)...")
    nn_index = NearestNeighbors(
        n_neighbors=min(n_candidates, len(train_embeds)),
        metric="cosine",
        algorithm="brute",
    )
    nn_index.fit(train_embeds)

    print(f"Scoring candidates with rerank head ({len(test_embeds)} queries)...")
    head.eval()
    with torch.no_grad():
        train_probs = F.softmax(
            head(torch.from_numpy(train_embeds).float()), dim=1,
        ).cpu().numpy()                                    # (n_train, 3)
        test_probs = F.softmax(
            head(torch.from_numpy(test_embeds).float()), dim=1,
        ).cpu().numpy()                                    # (n_test, 3)
        test_pred_class = test_probs.argmax(axis=1)         # (n_test,)

    # Compute label counts excluding self (we're retrieving from train for a
    # test query, so no self-match to subtract).
    unique, counts = np.unique(train_labels, return_counts=True)
    label_to_count = dict(zip(unique, counts))

    distances, indices = nn_index.kneighbors(test_embeds, n_neighbors=n_candidates)

    knn_acc = {k: {"precision": [], "recall": []} for k in ks}
    rr_acc = {k: {"precision": [], "recall": []} for k in ks}

    for q in range(len(test_embeds)):
        cand_idx = indices[q]
        cand_labels = train_labels[cand_idx]
        query_label = test_labels[q]
        total_relevant = label_to_count.get(query_label, 0)  # no self in train

        # Pure kNN ranking (already ordered by cosine distance asc)
        for k in ks:
            p, r = precision_recall_at_k(cand_labels, query_label, total_relevant, k)
            knn_acc[k]["precision"].append(p)
            knn_acc[k]["recall"].append(r)

        # Reranked: sort candidates by P(query's predicted class) — i.e.
        # how likely each candidate shares the query's predicted CTR class.
        rr_scores = train_probs[cand_idx, test_pred_class[q]]
        order = np.argsort(-rr_scores)
        reranked_labels = cand_labels[order]
        for k in ks:
            p, r = precision_recall_at_k(reranked_labels, query_label, total_relevant, k)
            rr_acc[k]["precision"].append(p)
            rr_acc[k]["recall"].append(r)

    def summarize(acc):
        return {
            k: {
                "precision_at_k": float(np.mean(acc[k]["precision"])),
                "recall_at_k":    float(np.mean(acc[k]["recall"])),
            }
            for k in ks
        }

    return summarize(knn_acc), summarize(rr_acc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an MLP rerank head on top of EfficientNet embeddings.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES)
    parser.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not EMBEDDINGS_PATH.exists() or not MAPPING_PATH.exists():
        raise FileNotFoundError(
            "Embeddings not found. Run scripts/extract_embeddings.py first."
        )

    embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
    mapping = pd.read_csv(MAPPING_PATH)
    if len(mapping) != embeddings.shape[0]:
        raise ValueError("embeddings / mapping row count mismatch")

    # Stratified 70/15/15 split on (CTR_label, niche)
    mapping = mapping.reset_index(drop=True)
    mapping["_strat_key"] = mapping["CTR_label"] + "_" + mapping["niche"].astype(str)
    idx_all = np.arange(len(mapping))

    idx_trainval, idx_test = train_test_split(
        idx_all, test_size=0.15, random_state=RANDOM_STATE,
        stratify=mapping["_strat_key"].to_numpy(),
    )
    val_ratio = 0.15 / 0.85
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_ratio, random_state=RANDOM_STATE,
        stratify=mapping.loc[idx_trainval, "_strat_key"].to_numpy(),
    )

    y_int = mapping["CTR_label"].map(LABEL_MAP).astype(np.int64).to_numpy()

    X_train_t = torch.from_numpy(embeddings[idx_train]).float()
    y_train_t = torch.from_numpy(y_int[idx_train]).long()
    X_val_t = torch.from_numpy(embeddings[idx_val]).float()
    y_val_t = torch.from_numpy(y_int[idx_val]).long()

    print(
        f"Splits — train={len(idx_train)} "
        f"val={len(idx_val)} test={len(idx_test)}"
    )

    # --- Train head -----------------------------------------------------
    HEAD_PATH.parent.mkdir(parents=True, exist_ok=True)
    head = RerankHead(in_dim=embeddings.shape[1], hidden=args.hidden, dropout=args.dropout)
    print(f"Training rerank head (hidden={args.hidden}, epochs={args.epochs})...")
    history = train_head(
        head, X_train_t, y_train_t, X_val_t, y_val_t,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
    )
    print(f"Best head saved to {HEAD_PATH}")

    # --- Retrieval + rerank eval ----------------------------------------
    train_labels = mapping.loc[idx_train, "CTR_label"].to_numpy()
    test_labels = mapping.loc[idx_test, "CTR_label"].to_numpy()

    knn_metrics, rr_metrics = evaluate_rerank(
        head,
        train_embeds=embeddings[idx_train],
        train_labels=train_labels,
        test_embeds=embeddings[idx_test],
        test_labels=test_labels,
        ks=args.ks,
        n_candidates=args.candidates,
    )

    out = {
        "num_test_queries": int(len(idx_test)),
        "ks": args.ks,
        "retrieval_candidates": args.candidates,
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "hidden": args.hidden,
            "dropout": args.dropout,
        },
        "history": history,
        "metrics": {
            "knn_only":  knn_metrics,
            "knn_plus_rerank": rr_metrics,
        },
        "relevance_rule": "retrieved item is relevant iff same CTR_label as query",
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(out, f, indent=2)

    # --- Report ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("Retrieve-then-rerank — test queries only (never seen by head)")
    print("=" * 60)
    for k in args.ks:
        k1 = knn_metrics[k]; k2 = rr_metrics[k]
        print(
            f"  K={k:<3d}  "
            f"kNN P@K={k1['precision_at_k']:.4f}  "
            f"-> rerank P@K={k2['precision_at_k']:.4f}   "
            f"(ΔP={k2['precision_at_k'] - k1['precision_at_k']:+.4f})"
        )
    print("=" * 60)
    print(f"\nMetrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()
