import csv
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET = PROJECT_ROOT / "results" / "processed" / "ml" / "traffic_prediction_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "processed" / "ml"

MODEL_FILE = OUTPUT_DIR / "traffic_congestion_model.joblib"
METRICS_FILE = OUTPUT_DIR / "traffic_congestion_metrics.csv"

DIRECTIONS = ["north", "south", "east", "west"]

NUMERIC_FEATURES = [
    "observed_north_queue",
    "observed_south_queue",
    "observed_east_queue",
    "observed_west_queue",
    "observed_north_waiting",
    "observed_south_waiting",
    "observed_east_waiting",
    "observed_west_waiting",
    "observed_signal_phase",
    "observed_vehicle_count",
    "observation_age",
]

CATEGORICAL_FEATURES = ["scenario", "intersection_id"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TRAIN_SEED = 1
VALIDATION_SEED = 2
TEST_SEED = 3


def make_pipeline(estimator):
    preprocessing = ColumnTransformer([
        ("numeric", "passthrough", NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         CATEGORICAL_FEATURES),
    ])

    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", estimator),
    ])


def metrics(actual, predicted, congestion_predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(np.asarray(predicted, dtype=float), 0)

    actual_congested = actual > 0
    nonzero_mae = (
        mean_absolute_error(
            actual[actual_congested],
            predicted[actual_congested],
        )
        if actual_congested.any() else float("nan")
    )

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "NonzeroMAE": nonzero_mae,
        "Precision": precision_score(
            actual_congested, congestion_predicted, zero_division=0
        ),
        "Recall": recall_score(
            actual_congested, congestion_predicted, zero_division=0
        ),
        "F1": f1_score(
            actual_congested, congestion_predicted, zero_division=0
        ),
    }


def choose_threshold(actual, probabilities):
    actual = np.asarray(actual) > 0

    candidates = []

    for threshold in np.arange(0.05, 0.96, 0.05):
        predicted = probabilities >= threshold

        p = precision_score(actual, predicted, zero_division=0)
        r = recall_score(actual, predicted, zero_division=0)
        f = f1_score(actual, predicted, zero_division=0)

        candidates.append((f, p, threshold))

    return max(candidates)[2]


def main():
    if not DATASET.exists():
        raise FileNotFoundError(DATASET)

    if MODEL_FILE.exists() or METRICS_FILE.exists():
        raise FileExistsError(
            "Congestion model outputs already exist. "
            "Refusing to overwrite."
        )

    df = pd.read_csv(DATASET)

    train = df[df["seed"] == TRAIN_SEED].copy()
    validation = df[df["seed"] == VALIDATION_SEED].copy()
    test = df[df["seed"] == TEST_SEED].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError("A required seed split is empty.")

    run_sets = [
        set(part["run_id"].unique())
        for part in [train, validation, test]
    ]

    if any(run_sets[i] & run_sets[j]
           for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Run overlap detected between splits.")

    print("TWO-STAGE CONGESTION PREDICTION")
    print("=" * 65)
    print(f"Training samples:   {len(train)}")
    print(f"Validation samples: {len(validation)}")
    print(f"Testing samples:    {len(test)}")
    print("Training seed: 1 | Validation seed: 2 | Test seed: 3")

    models = {}
    thresholds = {}
    result_rows = []

    for direction in DIRECTIONS:
        target = f"target_{direction}_queue"

        y_train = train[target].to_numpy(dtype=float)
        y_validation = validation[target].to_numpy(dtype=float)
        y_test = test[target].to_numpy(dtype=float)

        x_train = train[FEATURES]
        x_validation = validation[FEATURES]
        x_test = test[FEATURES]

        labels_train = (y_train > 0).astype(int)

        if len(np.unique(labels_train)) < 2:
            raise ValueError(f"Training data lacks both classes: {direction}")

        classifier = make_pipeline(
            RandomForestClassifier(
                n_estimators=150,
                max_depth=16,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            )
        )

        classifier.fit(x_train, labels_train)

        positive_index = list(classifier.classes_).index(1)
        validation_probabilities = classifier.predict_proba(
            x_validation
        )[:, positive_index]

        threshold = choose_threshold(
            y_validation, validation_probabilities
        )

        positive_train = y_train > 0

        regressor = make_pipeline(
            RandomForestRegressor(
                n_estimators=150,
                max_depth=16,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            )
        )

        regressor.fit(
            x_train.loc[positive_train],
            y_train[positive_train],
        )

        test_probabilities = classifier.predict_proba(
            x_test
        )[:, positive_index]

        congestion_predicted = test_probabilities >= threshold

        queue_magnitude = np.maximum(
            regressor.predict(x_test), 0
        )

        final_prediction = np.where(
            congestion_predicted, queue_magnitude, 0
        )

        zero_prediction = np.zeros(len(y_test))

        persistence_prediction = test[
            f"observed_{direction}_queue"
        ].to_numpy(dtype=float)

        for model_name, prediction, predicted_congestion in [
            (
                "Zero",
                zero_prediction,
                np.zeros(len(y_test), dtype=bool),
            ),
            (
                "Persistence",
                persistence_prediction,
                persistence_prediction > 0,
            ),
            (
                "TwoStage",
                final_prediction,
                congestion_predicted,
            ),
        ]:
            result = metrics(
                y_test, prediction, predicted_congestion
            )

            result_rows.append({
                "Direction": direction,
                "Model": model_name,
                "Samples": len(y_test),
                "NonzeroTargets": int((y_test > 0).sum()),
                "Threshold": (
                    round(float(threshold), 4)
                    if model_name == "TwoStage" else ""
                ),
                **{
                    name: round(float(value), 6)
                    for name, value in result.items()
                },
            })

        models[direction] = {
            "classifier": classifier,
            "regressor": regressor,
        }
        thresholds[direction] = float(threshold)

        print(
            f"\n{direction.upper()}: "
            f"Validation-selected threshold={threshold:.2f}"
        )

    pd.DataFrame(result_rows).to_csv(
        METRICS_FILE,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )

    joblib.dump({
        "models": models,
        "thresholds": thresholds,
        "features": FEATURES,
        "targets": [f"target_{d}_queue" for d in DIRECTIONS],
        "training_seed": TRAIN_SEED,
        "validation_seed": VALIDATION_SEED,
        "test_seed": TEST_SEED,
        "prediction_horizon_seconds": 10,
    }, MODEL_FILE)

    print("\n" + "=" * 65)
    print("TWO-STAGE MODEL TRAINING COMPLETE")
    print("Model:", MODEL_FILE)
    print("Metrics:", METRICS_FILE)
    print("=" * 65)


if __name__ == "__main__":
    main()
