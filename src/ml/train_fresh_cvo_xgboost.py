from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path("results/processed/ml/learned_cvo_fresh_queue_dataset.csv")
OUTPUT_DIR = Path("results/processed/ml")

MODEL_PATH = OUTPUT_DIR / "learned_cvo_fresh_queue_xgboost.joblib"
METRICS_PATH = OUTPUT_DIR / "learned_cvo_fresh_queue_xgboost_metrics.csv"
IMPORTANCE_PATH = OUTPUT_DIR / "learned_cvo_fresh_queue_xgboost_importance.csv"

NUMERIC_FEATURES = [
    "current_north_queue",
    "current_south_queue",
    "current_east_queue",
    "current_west_queue",
    "current_north_waiting",
    "current_south_waiting",
    "current_east_waiting",
    "current_west_waiting",
    "current_vehicle_count",
    "belief_age",
    "decision_margin",
]

CATEGORICAL_FEATURES = [
    "current_current_phase",
    "scenario",
    "intersection_id",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "fresh_cvo"


def evaluate(y_true, probabilities, threshold, name):
    probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "model": name,
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
        "brier_score": brier_score_loss(y_true, probabilities),
        "log_loss": log_loss(y_true, probabilities, labels=[0, 1]),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
    }


def main():
    for path in [MODEL_PATH, METRICS_PATH, IMPORTANCE_PATH]:
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing output: {path}"
            )

    df = pd.read_csv(DATA_PATH)
    df["seed"] = pd.to_numeric(df["seed"], errors="raise")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="raise").astype(int)

    required = set(FEATURES + ["seed", TARGET])
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Missing model values found.")

    train = df[df.seed == 1].copy()
    validation = df[df.seed == 2].copy()
    test = df[df.seed == 3].copy()

    if any(split.empty for split in [train, validation, test]):
        raise ValueError("One or more seed splits are empty.")

    for name, split in [
        ("training", train),
        ("validation", validation),
        ("test", test),
    ]:
        if set(split[TARGET].unique()) != {0, 1}:
            raise ValueError(f"{name} split must contain both classes.")

    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = validation[FEATURES], validation[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    scale_pos_weight = (
        (y_train == 0).sum() / (y_train == 1).sum()
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    classifier = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        reg_alpha=0.1,
        scale_pos_weight=float(scale_pos_weight),
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", classifier),
    ])

    print("Training fresh-queue XGBoost classifier...")
    print(f"Training positive-class weight: {scale_pos_weight:.4f}")

    model.fit(X_train, y_train)

    # Select the threshold only on validation Seed 2.
    val_probabilities = model.predict_proba(X_val)[:, 1]
    threshold_results = []

    for threshold in np.arange(0.05, 0.951, 0.01):
        predictions = (val_probabilities >= threshold).astype(int)

        threshold_results.append({
            "threshold": float(threshold),
            "f1": f1_score(y_val, predictions, zero_division=0),
            "precision": precision_score(
                y_val, predictions, zero_division=0
            ),
        })

    best = max(
        threshold_results,
        key=lambda item: (item["f1"], item["precision"]),
    )
    best_threshold = best["threshold"]

    # Final evaluation on unseen Seed 3.
    test_probabilities = model.predict_proba(X_test)[:, 1]

    metrics = [
        evaluate(
            y_test,
            test_probabilities,
            0.50,
            "xgboost_threshold_0.50",
        ),
        evaluate(
            y_test,
            test_probabilities,
            best_threshold,
            "xgboost_validation_selected",
        ),
    ]

    # Always-negative reference baseline.
    zero_probabilities = np.zeros(len(y_test), dtype=float)
    zero_predictions = np.zeros(len(y_test), dtype=int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, zero_predictions, labels=[0, 1]
    ).ravel()

    metrics.append({
        "model": "always_predict_zero",
        "threshold": 0.50,
        "accuracy": accuracy_score(y_test, zero_predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_test, zero_predictions
        ),
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "roc_auc": np.nan,
        "average_precision": np.nan,
        "brier_score": brier_score_loss(y_test, zero_probabilities),
        "log_loss": np.nan,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
    })

    metrics_df = pd.DataFrame(metrics)

    feature_names = model.named_steps[
        "preprocessing"
    ].get_feature_names_out()

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.named_steps[
            "classifier"
        ].feature_importances_,
    }).sort_values("importance", ascending=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(METRICS_PATH, index=False)
    importance_df.to_csv(IMPORTANCE_PATH, index=False)

    joblib.dump(
        {
            "model": model,
            "threshold": best_threshold,
            "features": FEATURES,
            "target": TARGET,
            "train_seed": 1,
            "validation_seed": 2,
            "test_seed": 3,
            "scale_pos_weight": float(scale_pos_weight),
        },
        MODEL_PATH,
    )

    print("\n" + "=" * 65)
    print("FRESH-QUEUE XGBOOST TRAINING COMPLETE")
    print("=" * 65)
    print(f"Training samples:   {len(train)}")
    print(f"Validation samples: {len(validation)}")
    print(f"Test samples:       {len(test)}")
    print(f"Validation F1:      {best['f1']:.4f}")
    print(f"Selected threshold: {best_threshold:.2f}")

    print("\nUnseen Seed 3 metrics:")
    print(metrics_df.to_string(index=False))

    print("\nTop 10 feature importances:")
    print(importance_df.head(10).to_string(index=False))

    print("\nSaved files:")
    print(MODEL_PATH)
    print(METRICS_PATH)
    print(IMPORTANCE_PATH)


if __name__ == "__main__":
    main()
