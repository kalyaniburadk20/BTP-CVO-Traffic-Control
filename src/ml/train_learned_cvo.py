from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path("results/processed/ml/learned_cvo_dataset.csv")
OUTPUT_DIR = Path("results/processed/ml")

MODEL_PATH = OUTPUT_DIR / "learned_cvo_random_forest.joblib"
METRICS_PATH = OUTPUT_DIR / "learned_cvo_metrics.csv"
IMPORTANCE_PATH = OUTPUT_DIR / "learned_cvo_feature_importance.csv"


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
    "hypothetical_north_queue",
    "hypothetical_south_queue",
    "hypothetical_east_queue",
    "hypothetical_west_queue",
    "hypothetical_north_waiting",
    "hypothetical_south_waiting",
    "hypothetical_east_waiting",
    "hypothetical_west_waiting",
    "hypothetical_vehicle_count",
    "belief_age",
    "decision_margin",
]

CATEGORICAL_FEATURES = [
    "scenario",
    "intersection_id",
    "current_current_phase",
    "hypothetical_current_phase",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def evaluate(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
        "brier_score": brier_score_loss(y_true, probabilities),
        "log_loss": log_loss(y_true, probabilities, labels=[0, 1]),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    existing = [
        path for path in [MODEL_PATH, METRICS_PATH, IMPORTANCE_PATH]
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing outputs: "
            + ", ".join(str(path) for path in existing)
        )

    df = pd.read_csv(DATA_PATH)
    df["seed"] = pd.to_numeric(df["seed"], errors="raise")

    required = set(FEATURES + ["seed", "cvo"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df[FEATURES + ["cvo"]].isna().any().any():
        raise ValueError("Missing values found in model data.")

    if not set(df["cvo"].unique()).issubset({0, 1}):
        raise ValueError("CVO label must contain only 0 and 1.")

    train = df[df["seed"] == 1]
    validation = df[df["seed"] == 2]
    test = df[df["seed"] == 3]

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("One or more seed splits are empty.")

    for split_name, split in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        if set(split["cvo"].unique()) != {0, 1}:
            raise ValueError(f"{split_name} split does not contain both classes.")

    X_train = train[FEATURES]
    y_train = train["cvo"].astype(int)

    X_val = validation[FEATURES]
    y_val = validation["cvo"].astype(int)

    X_test = test[FEATURES]
    y_test = test["cvo"].astype(int)

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

    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=16,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("Training Learned CVO Random Forest...")
    model.fit(X_train, y_train)

    validation_probabilities = model.predict_proba(X_val)[:, 1]

    thresholds = np.arange(0.10, 0.91, 0.01)
    threshold_results = [
        (float(threshold), f1_score(
            y_val,
            (validation_probabilities >= threshold).astype(int),
            zero_division=0,
        ))
        for threshold in thresholds
    ]

    best_threshold, best_validation_f1 = max(
        threshold_results,
        key=lambda item: (item[1], -abs(item[0] - 0.5)),
    )

    test_probabilities = model.predict_proba(X_test)[:, 1]

    metrics = []
    for threshold_name, threshold in [
        ("default_0.50", 0.50),
        ("validation_selected", best_threshold),
    ]:
        result = evaluate(y_test, test_probabilities, threshold)
        result["split"] = "test_seed_3"
        result["threshold_selection"] = threshold_name
        metrics.append(result)

    all_negative_predictions = np.zeros(len(y_test), dtype=int)
    baseline_probabilities = np.zeros(len(y_test), dtype=float)

    baseline = {
        "split": "test_seed_3",
        "threshold_selection": "always_predict_zero",
        "threshold": 0.50,
        "accuracy": accuracy_score(y_test, all_negative_predictions),
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "roc_auc": np.nan,
        "average_precision": np.nan,
        "brier_score": brier_score_loss(y_test, baseline_probabilities),
        "log_loss": np.nan,
    }
    metrics.append(baseline)

    importance = model.named_steps["classifier"].feature_importances_
    feature_names = model.named_steps[
        "preprocessing"
    ].get_feature_names_out()

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(METRICS_PATH, index=False)
    importance_df.to_csv(IMPORTANCE_PATH, index=False)

    joblib.dump(
        {
            "model": model,
            "threshold": best_threshold,
            "features": FEATURES,
            "target": "cvo",
            "train_seed": 1,
            "validation_seed": 2,
            "test_seed": 3,
        },
        MODEL_PATH,
    )

    print("\n" + "=" * 65)
    print("LEARNED CVO TRAINING COMPLETE")
    print("=" * 65)
    print(f"Training samples:   {len(train)}")
    print(f"Validation samples: {len(validation)}")
    print(f"Test samples:       {len(test)}")
    print(f"Input features:     {len(FEATURES)}")
    print(f"Validation F1:      {best_validation_f1:.4f}")
    print(f"Selected threshold: {best_threshold:.2f}")

    print("\nTest metrics:")
    print(metrics_df.to_string(index=False))

    print("\nTop 10 feature importances:")
    print(importance_df.head(10).to_string(index=False))

    print("\nSaved files:")
    print(MODEL_PATH)
    print(METRICS_PATH)
    print(IMPORTANCE_PATH)


if __name__ == "__main__":
    main()
