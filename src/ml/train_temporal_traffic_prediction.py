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
DATASET = (
    PROJECT_ROOT / "results" / "processed" / "ml"
    / "traffic_prediction_dataset_temporal_v2.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "processed" / "ml"
MODEL_FILE = OUTPUT_DIR / "traffic_prediction_temporal_rf.joblib"
METRICS_FILE = OUTPUT_DIR / "traffic_prediction_temporal_rf_metrics.csv"

DIRECTIONS = ["north", "south", "east", "west"]

NUMERIC_FEATURES = [
    "observed_north_queue", "observed_south_queue",
    "observed_east_queue", "observed_west_queue",
    "observed_north_waiting", "observed_south_waiting",
    "observed_east_waiting", "observed_west_waiting",
    "observed_signal_phase", "observed_vehicle_count",

    "queue_change_north_per_second",
    "queue_change_south_per_second",
    "queue_change_east_per_second",
    "queue_change_west_per_second",

    "waiting_change_north_per_second",
    "waiting_change_south_per_second",
    "waiting_change_east_per_second",
    "waiting_change_west_per_second",

    "observation_age",
    "previous_observation_interval",
    "has_previous_observation",
]

CATEGORICAL_FEATURES = [
    "scenario",
    "strategy",
    "intersection_id",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGETS = [
    f"target_{direction}_queue"
    for direction in DIRECTIONS
]


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(
        np.asarray(predicted, dtype=float), 0
    )

    nonzero = actual > 0
    predicted_congested = predicted >= 0.5

    nonzero_mae = (
        mean_absolute_error(
            actual[nonzero],
            predicted[nonzero],
        )
        if nonzero.any()
        else float("nan")
    )

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "NonzeroMAE": nonzero_mae,
        "Precision": precision_score(
            nonzero, predicted_congested, zero_division=0
        ),
        "Recall": recall_score(
            nonzero, predicted_congested, zero_division=0
        ),
        "F1": f1_score(
            nonzero, predicted_congested, zero_division=0
        ),
    }


def main():
    if not DATASET.exists():
        raise FileNotFoundError(DATASET)

    if MODEL_FILE.exists() or METRICS_FILE.exists():
        raise FileExistsError(
            "Temporal RF output already exists. "
            "Refusing to overwrite existing files."
        )

    df = pd.read_csv(DATASET)

    required = FEATURES + TARGETS + [
        "run_id", "seed", "time"
    ]

    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    train = df[df["seed"].isin([1, 2])].copy()
    test = df[df["seed"] == 3].copy()

    if train.empty or test.empty:
        raise ValueError("Training or testing data is empty.")

    train_runs = set(train["run_id"].unique())
    test_runs = set(test["run_id"].unique())

    if train_runs.intersection(test_runs):
        raise ValueError("Training and testing runs overlap.")

    preprocessing = ColumnTransformer([
        ("numeric", "passthrough", NUMERIC_FEATURES),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL_FEATURES,
        ),
    ])

    model = Pipeline([
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
    ])

    print("=" * 70)
    print("TEMPORAL RANDOM FOREST TRAFFIC PREDICTION")
    print("=" * 70)
    print(f"Training samples: {len(train)}")
    print(f"Testing samples:  {len(test)}")
    print(f"Training runs:    {len(train_runs)}")
    print(f"Testing runs:     {len(test_runs)}")
    print("Training seeds: 1, 2")
    print("Testing seed:   3")
    print(f"Input features:  {len(FEATURES)}")
    print(f"Queue targets:  {len(TARGETS)}")

    print("\nTraining temporal Random Forest...")
    model.fit(train[FEATURES], train[TARGETS])

    predictions = np.maximum(
        model.predict(test[FEATURES]), 0
    )

    results = []

    print("\nHELD-OUT SEED 3 RESULTS")
    print("=" * 70)

    for scenario in ["low", "medium", "high"]:

        mask = test["scenario"].eq(scenario).to_numpy()
        scenario_data = test.loc[mask]

        print(f"\n{scenario.upper()} TRAFFIC")

        for index, direction in enumerate(DIRECTIONS):

            actual = scenario_data[
                f"target_{direction}_queue"
            ].to_numpy(dtype=float)

            temporal_prediction = predictions[mask, index]

            zero_prediction = np.zeros(len(actual))

            persistence_prediction = scenario_data[
                f"observed_{direction}_queue"
            ].to_numpy(dtype=float)

            candidates = [
                ("Zero", zero_prediction),
                ("Persistence", persistence_prediction),
                ("TemporalRF", temporal_prediction),
            ]

            row = {
                "Scenario": scenario,
                "Direction": direction,
                "Samples": len(actual),
                "NonzeroTargets": int((actual > 0).sum()),
            }

            for name, prediction in candidates:

                values = calculate_metrics(
                    actual, prediction
                )

                for metric, value in values.items():
                    row[f"{name}_{metric}"] = round(
                        float(value), 6
                    )

            results.append(row)

            print(
                f"{direction.upper():<6} "
                f"Zero MAE={row['Zero_MAE']:.4f} | "
                f"Persistence MAE={row['Persistence_MAE']:.4f} | "
                f"Temporal RF MAE={row['TemporalRF_MAE']:.4f} | "
                f"RF F1={row['TemporalRF_F1']:.4f}"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(results).to_csv(
        METRICS_FILE, index=False
    )

    joblib.dump({
        "model": model,
        "features": FEATURES,
        "targets": TARGETS,
        "training_seeds": [1, 2],
        "test_seed": 3,
        "prediction_horizon_seconds": 10,
    }, MODEL_FILE)

    print("\n" + "=" * 70)
    print("TEMPORAL RANDOM FOREST TRAINING COMPLETE")
    print("Model:", MODEL_FILE)
    print("Metrics:", METRICS_FILE)
    print("=" * 70)


if __name__ == "__main__":
    main()
