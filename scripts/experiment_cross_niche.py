"""
Cross-niche generalization experiment.

Research question: Does "good thumbnail design" transfer across content niches?

Method:
- For each source niche, train XGBoost on ALL data from that niche.
- Test on ALL data from every OTHER niche (and on a held-out split from
  the same niche as a reference).
- Also train on ALL niches combined (80/20 split) as a control baseline.
- Report accuracy and macro-F1 for every train->test combination.
- Save a performance matrix CSV and a seaborn heatmap.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "features.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"
MATRIX_CSV = OUTPUT_DIR / "cross_niche_matrix.csv"
HEATMAP_PNG = OUTPUT_DIR / "cross_niche_heatmap.png"

RANDOM_STATE = 42

# Feature columns (same as train_xgboost.py)
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
# XGBoost helper
# ---------------------------------------------------------------------------
def make_xgb() -> XGBClassifier:
    """Return an XGBClassifier with tuned defaults."""
    return XGBClassifier(
        max_depth=5,
        learning_rate=0.1,
        n_estimators=200,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        verbosity=0,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # Load data
    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded {len(df)} samples from {FEATURES_CSV}")

    # Encode labels: Low=0, Medium=1, High=2
    le = LabelEncoder()
    le.classes_ = np.array(["Low", "Medium", "High"])
    df["label_enc"] = le.transform(df["label"])

    niches = sorted(df["niche"].unique())
    print(f"Niches found: {niches}")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}\n")

    # ------------------------------------------------------------------
    # Results storage
    # ------------------------------------------------------------------
    # rows = training niche (or "All"), cols = test niche (or "All")
    row_labels = niches + ["All"]
    col_labels = niches + ["All"]
    acc_matrix = pd.DataFrame(
        np.nan, index=row_labels, columns=col_labels, dtype=float
    )
    f1_matrix = pd.DataFrame(
        np.nan, index=row_labels, columns=col_labels, dtype=float
    )

    # ------------------------------------------------------------------
    # Per-niche training -> cross-niche testing
    # ------------------------------------------------------------------
    for src_niche in niches:
        src_df = df[df["niche"] == src_niche].copy()
        X_src = src_df[FEATURE_COLS].values
        y_src = src_df["label_enc"].values

        # Same-niche evaluation: 80/20 hold-out
        X_train, X_test_same, y_train, y_test_same = train_test_split(
            X_src, y_src, test_size=0.20, random_state=RANDOM_STATE, stratify=y_src
        )

        clf = make_xgb()
        clf.fit(X_train, y_train)

        # Same-niche test
        y_pred_same = clf.predict(X_test_same)
        acc_matrix.loc[src_niche, src_niche] = accuracy_score(y_test_same, y_pred_same)
        f1_matrix.loc[src_niche, src_niche] = f1_score(
            y_test_same, y_pred_same, average="macro", zero_division=0
        )

        # Now retrain on ALL source data for cross-niche tests
        clf_full = make_xgb()
        clf_full.fit(X_src, y_src)

        for tgt_niche in niches:
            if tgt_niche == src_niche:
                continue  # already handled above with hold-out
            tgt_df = df[df["niche"] == tgt_niche]
            X_tgt = tgt_df[FEATURE_COLS].values
            y_tgt = tgt_df["label_enc"].values

            y_pred = clf_full.predict(X_tgt)
            acc_matrix.loc[src_niche, tgt_niche] = accuracy_score(y_tgt, y_pred)
            f1_matrix.loc[src_niche, tgt_niche] = f1_score(
                y_tgt, y_pred, average="macro", zero_division=0
            )

    # ------------------------------------------------------------------
    # "All" control: train on combined data with 80/20 split
    # ------------------------------------------------------------------
    X_all = df[FEATURE_COLS].values
    y_all = df["label_enc"].values
    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
        X_all, y_all, test_size=0.20, random_state=RANDOM_STATE, stratify=y_all
    )

    clf_all = make_xgb()
    clf_all.fit(X_train_all, y_train_all)

    y_pred_all = clf_all.predict(X_test_all)
    acc_matrix.loc["All", "All"] = accuracy_score(y_test_all, y_pred_all)
    f1_matrix.loc["All", "All"] = f1_score(
        y_test_all, y_pred_all, average="macro", zero_division=0
    )

    # Also evaluate "All" model on each individual niche (full niche data)
    for niche in niches:
        niche_df = df[df["niche"] == niche]
        X_n = niche_df[FEATURE_COLS].values
        y_n = niche_df["label_enc"].values
        y_pred_n = clf_all.predict(X_n)
        acc_matrix.loc["All", niche] = accuracy_score(y_n, y_pred_n)
        f1_matrix.loc["All", niche] = f1_score(
            y_n, y_pred_n, average="macro", zero_division=0
        )

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print("=" * 70)
    print("CROSS-NICHE GENERALIZATION — Accuracy Matrix")
    print("(rows = train niche, columns = test niche)")
    print("=" * 70)
    print(acc_matrix.round(4).to_string())

    print()
    print("=" * 70)
    print("CROSS-NICHE GENERALIZATION — F1 (macro) Matrix")
    print("(rows = train niche, columns = test niche)")
    print("=" * 70)
    print(f1_matrix.round(4).to_string())

    # ------------------------------------------------------------------
    # Save CSV (combine acc & f1 into one file)
    # ------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build a combined CSV with a metric column
    acc_flat = acc_matrix.stack().reset_index()
    acc_flat.columns = ["train_niche", "test_niche", "value"]
    acc_flat["metric"] = "accuracy"

    f1_flat = f1_matrix.stack().reset_index()
    f1_flat.columns = ["train_niche", "test_niche", "value"]
    f1_flat["metric"] = "f1_macro"

    combined = pd.concat([acc_flat, f1_flat], ignore_index=True)
    combined = combined[["metric", "train_niche", "test_niche", "value"]]
    combined.to_csv(MATRIX_CSV, index=False)
    print(f"\nMatrix saved to {MATRIX_CSV}")

    # ------------------------------------------------------------------
    # Heatmap visualization
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, matrix, title in [
        (axes[0], acc_matrix, "Accuracy"),
        (axes[1], f1_matrix, "F1 (macro)"),
    ]:
        sns.heatmap(
            matrix.astype(float),
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            vmin=0.0,
            vmax=1.0,
            linewidths=0.5,
            ax=ax,
            cbar_kws={"label": title},
        )
        ax.set_title(f"Cross-Niche {title}")
        ax.set_xlabel("Test Niche")
        ax.set_ylabel("Train Niche")

    plt.suptitle(
        "Cross-Niche Generalization: Does good thumbnail design transfer?",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(HEATMAP_PNG, dpi=150, bbox_inches="tight")
    print(f"Heatmap saved to {HEATMAP_PNG}")
    plt.close(fig)


if __name__ == "__main__":
    main()
