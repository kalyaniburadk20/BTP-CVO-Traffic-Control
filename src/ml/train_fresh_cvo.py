from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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


DATA_PATH = Path(
    "results/processed/ml/learned_cvo_fresh_queue_dataset.csv"
)
OUTPUT_DIR = Path("results/processed/ml")

MODEL_PATH = OUTPUT_DIR / "learned_cvo_fresh_queue_random_forest.joblib"
METRICS_PATH = OUTPUT_DIR / "learned_cvo_fresh_queue_metrics.csv"
IMPORTANCE_PATH = OUTPUT_DIR / "learned_cvo_fresh_queue_feature_importance.csv"


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


def evaluate_model(y_true, probabilities, threshold, name):
    safe_probabilities = np.clip(
        probabilities, 1e-7, 1 - 1e-7
    )
    predictions = (
        safe_probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "model": name,
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, predictions
        ),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_true, safe_probabilities
        ),
        "average_precision": average_precision_score(
            y_true, safe_probabilities
        ),
        "brier_score": brier_score_loss(
            y_true, safe_probabilities
        ),
        "log_loss": log_loss(
            y_true, safe_probabilities, labels=[0, 1]
        ),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
        "actual_positive": int(np.sum(y_true == 1)),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    existing_outputs = [
        path
        for path in [
            MODEL_PATH,
            METRICS_PATH,
            IMPORTANCE_PATH,
        ]
        if path.exists()
    ]

    if existing_outputs:
        raise FileExistsError(
            "Refusing to overwrite existing outputs: "
            + ", ".join(str(path) for path in existing_outputs)
        )

    df = pd.read_csv(DATA_PATH)
    df["seed"] = pd.to_numeric(df["seed"], errors="raise")
    df[TARGET] = pd.to_numeric(
        df[TARGET], errors="raise"
    ).astype(int)

    required_columns = set(
        FEATURES + [TARGET, "seed"]
    )
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    if df[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Missing values found.")

    if set(df[TARGET].unique()) != {0, 1}:
        raise ValueError("Target must contain both classes.")

    train = df[df["seed"] == 1].copy()
    validation = df[df["seed"] == 2].copy()
    test = df[df["seed"] == 3].copy()

    for name, split in [
        ("Training", train),
        ("Validation", validation),
        ("Test", test),
    ]:
        if split.empty or set(split[TARGET].unique()) != {0, 1}:
            raise ValueError(
                f"{name} split is empty or missing a class."
            )

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_val = validation[FEATURES]
    y_val = validation[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=16,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("Training fresh-queue Learned CVO classifier...")

    model.fit(X_train, y_train)

    # Threshold selection uses validation data only.
    val_probabilities = model.predict_proba(X_val)[:, 1]

    threshold_results = []

    for threshold in np.arange(0.05, 0.951, 0.01):
        predictions = (
            val_probabilities >= threshold
        ).astype(int)

        threshold_results.append({
            "threshold": float(threshold),
            "f1": f1_score(
                y_val, predictions, zero_division=0
            ),
            "precision": precision_score(
                y_val, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_val, predictions, zero_division=0
            ),
        })

    best_result = max(
        threshold_results,
        key=lambda item: (
            item["f1"],
            item["precision"],
        ),
    )

    best_threshold = best_result["threshold"]

    # Final evaluation on untouched seed 3.
    test_probabilities = model.predict_proba(X_test)[:, 1]

    metrics = [
        evaluate_model(
            y_test,
            test_probabilities,
            0.50,
            "random_forest_threshold_0.50",
        ),
        evaluate_model(
            y_test,
            test_probabilities,
            best_threshold,
            "random_forest_validation_selected",
        ),
    ]

    # Majority-class reference: always predict no action change.
    zero_predictions = np.zeros(len(y_test), dtype=int)
    zero_probabilities = np.zeros(len(y_test), dtype=float)

    tn, fp, fn, tp = confusion_matrix(
        y_test, zero_predictions, labels=[0, 1]
    ).ravel()

    baseline = {
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
        "brier_score": brier_score_loss(
            y_test, zero_probabilities
        ),
        "log_loss": np.nan,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
        "actual_positive": int(np.sum(y_test == 1)),
    }

    metrics.append(baseline)

    metrics_df = pd.DataFrame(metrics)

    # Feature importance after preprocessing.
    classifier = model.named_steps["classifier"]
    feature_names = model.named_steps[
        "preprocessing"
    ].get_feature_names_out()

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": classifier.feature_importances_,
    }).sort_values(
        "importance", ascending=False
    )

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
            "label_definition": (
                "Action divergence between the current belief "
                "and actual fresh directional queues, using "
                "the same pre-acquisition signal-phase context."
            ),
        },
        MODEL_PATH,
    )

    print("\n" + "=" * 65)
    print("FRESH-QUEUE LEARNED CVO TRAINING COMPLETE")
    print("=" * 65)
    print(f"Training samples:   {len(train)}")
    print(f"Validation samples: {len(validation)}")
    print(f"Test samples:       {len(test)}")
    print(f"Input features:     {len(FEATURES)}")
    print(f"Training positives: {int(y_train.sum())}")
    print(f"Validation positives: {int(y_val.sum())}")
    print(f"Test positives: {int(y_test.sum())}")

    print("\nValidation threshold selection:")
    print(f"Selected threshold: {best_threshold:.2f}")
    print(f"Validation F1:      {best_result['f1']:.4f}")
    print(f"Validation precision: {best_result['precision']:.4f}")
    print(f"Validation recall:    {best_result['recall']:.4f}")

    print("\nUnseen Seed 3 test metrics:")
    print(metrics_df.to_string(index=False))

    print("\nTop 10 feature importances:")
    print(importance_df.head(10).to_string(index=False))

    print("\nSaved files:")
    print(MODEL_PATH)
    print(METRICS_PATH)
    print(IMPORTANCE_PATH)


if __name__ == "__main__":
    main()
