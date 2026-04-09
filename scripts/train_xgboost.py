"""
XGBoost classifier trained on handcrafted thumbnail features.

Performs grid-search cross-validation, evaluates on a held-out test set,
saves the trained model / label encoder, and exports a feature-importance
bar chart.
"""

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "features.csv"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"

RANDOM_STATE = 42

# Feature columns (all numeric thumbnail features)
FEATURE_COLS = [
    "dominant_color_1_r", "dominant_color_1_g", "dominant_color_1_b",
    "dominant_color_2_r", "dominant_color_2_g", "dominant_color_2_b",
    "dominant_color_3_r", "dominant_color_3_g", "dominant_color_3_b",
    "brightness", "contrast", "saturation",
    "face_count", "face_area_ratio",
    "text_present", "text_area_ratio",
    "edge_density", "color_entropy",
]


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load & encode
    # ------------------------------------------------------------------
    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded {len(df)} samples from {FEATURES_CSV}")

    le = LabelEncoder()
    le.classes_ = np.array(["Low", "Medium", "High"])  # 0, 1, 2
    df["label_enc"] = le.transform(df["label"])

    X = df[FEATURE_COLS].values
    y = df["label_enc"].values

    # ------------------------------------------------------------------
    # 2. Stratified train / val / test  (70 / 15 / 15)
    # ------------------------------------------------------------------
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y,
    )
    # From the remaining 85 %, carve out 15 / 85 ≈ 17.65 % for val
    val_ratio = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_ratio,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    # ------------------------------------------------------------------
    # 3. Grid-search CV on training data
    # ------------------------------------------------------------------
    param_grid = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.3],
        "n_estimators": [100, 200],
        "subsample": [0.8, 1.0],
    }

    base_model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        verbosity=0,
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )
    print("\nRunning grid search (this may take a few minutes) ...")
    grid_search.fit(X_train, y_train)

    print(f"\nBest params: {grid_search.best_params_}")
    print(f"Best CV F1 (macro): {grid_search.best_score_:.4f}")

    # ------------------------------------------------------------------
    # 4. Train final model with best params on full training set
    # ------------------------------------------------------------------
    best_params = grid_search.best_params_
    final_model = XGBClassifier(
        **best_params,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    final_model.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 5. Evaluate on test set
    # ------------------------------------------------------------------
    y_pred = final_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")

    print("\n" + "=" * 50)
    print("XGBoost – Test Metrics")
    print("=" * 50)
    print(f"  Accuracy:          {accuracy:.4f}")
    print(f"  F1 (macro):        {f1:.4f}")
    print(f"  Precision (macro): {precision:.4f}")
    print(f"  Recall (macro):    {recall:.4f}")
    print("=" * 50)

    # ------------------------------------------------------------------
    # 6. Feature importances chart
    # ------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    importances = final_model.feature_importances_
    sorted_idx = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        [FEATURE_COLS[i] for i in sorted_idx],
        importances[sorted_idx],
        color="steelblue",
    )
    ax.set_xlabel("Feature Importance")
    ax.set_title("XGBoost Feature Importances")
    plt.tight_layout()
    importance_path = OUTPUT_DIR / "xgboost_feature_importances.png"
    fig.savefig(importance_path, dpi=150)
    plt.close(fig)
    print(f"\nFeature importance chart saved to {importance_path}")

    # ------------------------------------------------------------------
    # 7. Persist model & label encoder
    # ------------------------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "xgboost_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    print(f"XGBoost model saved to {model_path}")

    le_path = MODEL_DIR / "label_encoder.pkl"
    with open(le_path, "wb") as f:
        pickle.dump(le, f)
    print(f"Label encoder saved to {le_path}")


if __name__ == "__main__":
    main()
