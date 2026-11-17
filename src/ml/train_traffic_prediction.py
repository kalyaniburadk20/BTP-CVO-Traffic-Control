import csv
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
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
DATASET_FILE = (
    PROJECT_ROOT / "results" / "processed" / "ml"
    / "traffic_prediction_dataset.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "processed" / "ml"

MODEL_FILE = OUTPUT_DIR / "traffic_prediction_random_forest.joblib"
METRICS_FILE = OUTPUT_DIR / "traffic_prediction_random_forest_metrics.csv"

QUEUE_DIRECTIONS = ["north", "south", "east", "west"]

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

CATEGORICAL_FEATURES = [
    "scenario",
    "strategy",
    "intersection_id",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGETS = [
    f"target_{direction}_queue"
    for direction in QUEUE_DIRECTIONS
]

TRAIN_SEEDS = [1, 2]
TEST_SEEDS = [3]


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    predicted = np.maximum(predicted, 0)

    nonzero_mask = actual > 0
    actual_positive = nonzero_mask.astype(int)
    predicted_positive = (predicted >= 0.5).astype(int)

    nonzero_mae = (
        mean_absolute_error(
            actual[nonzero_mask],
            predicted[nonzero_mask],
        )
        if nonzero_mask.any()
        else float("nan")
    )

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "NonzeroMAE": nonzero_mae,
        "CongestionPrecision": precision_score(
            actual_positive,
            predicted_positive,
            zero_division=0,
        ),
        "CongestionRecall": recall_score(
            actual_positive,
            predicted_positive,
            zero_division=0,
        ),
        "CongestionF1": f1_score(
            actual_positive,
            predicted_positive,
            zero_division=0,
        ),
    }


def main():
    if not DATASET_FILE.exists():
        raise FileNotFoundError(DATASET_FILE)

    if MODEL_FILE.exists() or METRICS_FILE.exists():
        raise FileExistsError(
            "Training outputs already exist. "
            "Refusing to overwrite existing files."
        )

    df = pd.read_csv(DATASET_FILE)

    required_columns = FEATURES + TARGETS + [
        "run_id", "seed", "scenario", "strategy",
        "intersection_id", "time",
    ]

    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")

    train = df[df["seed"].isin(TRAIN_SEEDS)].copy()
    test = df[df["seed"].isin(TEST_SEEDS)].copy()

    if train.empty or test.empty:
        raise ValueError("Training or testing split is empty.")

    train_runs = set(train["run_id"].unique())
    test_runs = set(test["run_id"].unique())

    if train_runs.intersection(test_runs):
        raise ValueError("Data leakage: run IDs overlap.")

    print("=" * 70)
    print("RANDOM FOREST TRAFFIC PREDICTION")
    print("=" * 70)
    print(f"Training samples: {len(train)}")
    print(f"Testing samples:  {len(test)}")
    print(f"Training runs:    {len(train_runs)}")
    print(f"Testing runs:     {len(test_runs)}")
    print("Training seeds:", TRAIN_SEEDS)
    print("Testing seeds:", TEST_SEEDS)

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
                "regressor",
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=16,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    X_train = train[FEATURES]
    y_train = train[TARGETS]

    X_test = test[FEATURES]
    y_test = test[TARGETS]

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    predictions = np.maximum(model.predict(X_test), 0)

    result_rows = []

    print("\nHELD-OUT SEED 3 RESULTS")
    print("=" * 70)

    for scenario in sorted(test["scenario"].unique()):
        scenario_mask = test["scenario"].eq(scenario).to_numpy()

        print(f"\n{scenario.upper()} TRAFFIC")

        for index, direction in enumerate(QUEUE_DIRECTIONS):
            actual = y_test[TARGETS[index]].to_numpy(dtype=float)[scenario_mask]
            predicted = predictions[scenario_mask, index]

            zero_pred = np.zeros(len(actual))
            persistence_pred = test.loc[
                scenario_mask,
                f"observed_{direction}_queue",
            ].to_numpy(dtype=float)

            model_metrics = calculate_metrics(actual, predicted)
            zero_metrics = calculate_metrics(actual, zero_pred)
            persistence_metrics = calculate_metrics(
                actual, persistence_pred
            )

            result = {
                "Scenario": scenario,
                "Direction": direction,
                "Samples": len(actual),
                "NonzeroTargets": int((actual > 0).sum()),
            }

            for name, values in [
                ("Zero", zero_metrics),
                ("Persistence", persistence_metrics),
                ("RandomForest", model_metrics),
            ]:
                for metric, value in values.items():
                    result[f"{name}_{metric}"] = round(float(value), 6)

            result_rows.append(result)

            print(
                f"{direction.upper():<6} "
                f"Zero MAE={zero_metrics['MAE']:.4f} | "
                f"Persistence MAE={persistence_metrics['MAE']:.4f} | "
                f"RF MAE={model_metrics['MAE']:.4f} | "
                f"RF RMSE={model_metrics['RMSE']:.4f} | "
                f"RF Recall={model_metrics['CongestionRecall']:.4f}"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(result_rows).to_csv(
        METRICS_FILE,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "targets": TARGETS,
            "training_seeds": TRAIN_SEEDS,
            "test_seeds": TEST_SEEDS,
            "prediction_horizon_seconds": 10,
        },
        MODEL_FILE,
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print(f"Model: {MODEL_FILE}")
    print(f"Metrics: {METRICS_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
