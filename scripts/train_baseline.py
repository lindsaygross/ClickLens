"""
Baseline model: Majority class classifier.

Loads the handcrafted features CSV, identifies the most common label in the
training split, and predicts that label for every test sample.  Saves the
majority class (and class distribution) as a lightweight pickle file.
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths (resolved relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "features.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "baseline_model.pkl"

RANDOM_STATE = 42


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded {len(df)} samples from {FEATURES_CSV}")
    print(f"Label distribution:\n{df['label'].value_counts()}\n")

    # ------------------------------------------------------------------
    # 2. Stratified train / test split  (70 / 30)
    # ------------------------------------------------------------------
    train_df, test_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    print(f"Train size: {len(train_df)}  |  Test size: {len(test_df)}")

    # ------------------------------------------------------------------
    # 3. Find the majority class in the training set
    # ------------------------------------------------------------------
    class_counts = train_df["label"].value_counts()
    majority_class = class_counts.idxmax()
    class_distribution = class_counts.to_dict()
    print(f"\nTraining class distribution:\n{class_counts}")
    print(f"Majority class: {majority_class}\n")

    # ------------------------------------------------------------------
    # 4. Predict majority class for all test samples
    # ------------------------------------------------------------------
    y_test = test_df["label"].values
    y_pred = [majority_class] * len(y_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)

    print("=" * 50)
    print("Baseline (Majority Class) – Test Metrics")
    print("=" * 50)
    print(f"  Accuracy:         {accuracy:.4f}")
    print(f"  F1 (macro):       {f1:.4f}")
    print(f"  Precision (macro):{precision:.4f}")
    print(f"  Recall (macro):   {recall:.4f}")
    print("=" * 50)

    # ------------------------------------------------------------------
    # 5. Persist the "model"
    # ------------------------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_artifact = {
        "majority_class": majority_class,
        "class_distribution": class_distribution,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_artifact, f)
    print(f"\nBaseline model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
