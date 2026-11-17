from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
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
MODEL_FILE = OUTPUT_DIR / "traffic_prediction_xgboost.joblib"
METRICS_FILE = OUTPUT_DIR / "traffic_prediction_xgboost_metrics.csv"

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


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(
        np.asarray(predicted, dtype=float), 0
    )

    actual_positive = actual > 0
    predicted_positive = predicted >= 0.5

    nonzero_mae = (
        mean_absolute_error(
            actual[actual_positive],
            predicted[actual_positive],
        )
        if actual_positive.any()
        else float("nan")
    )

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "NonzeroMAE": nonzero_mae,
        "Precision": precision_score(
            actual_positive, predicted_positive,
            zero_division=0,
        ),
        "Recall": recall_score(
            actual_positive, predicted_positive,
            zero_division=0,
        ),
        "F1": f1_score(
            actual_positive, predicted_positive,
            zero_division=0,
        ),
    }


def main():
    if not DATASET.exists():
        raise FileNotFoundError(DATASET)

    if MODEL_FILE.exists() or METRICS_FILE.exists():
        raise FileExistsError(
            "XGBoost output already exists. "
            "Refusing to overwrite existing files."
        )

    df = pd.read_csv(DATASET)

    targets = [
        f"target_{direction}_queue"
        for direction in DIRECTIONS
    ]

    required = FEATURES + targets + ["run_id", "seed", "time"]
    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")

    train = df[df["seed"].isin([1, 2])].copy()
    test = df[df["seed"] == 3].copy()

    if train.empty or test.empty:
        raise ValueError("Training or testing data is empty.")

    if set(train["run_id"]).intersection(set(test["run_id"])):
        raise ValueError("Training and testing runs overlap.")

    print("=" * 70)
    print("XGBOOST TEMPORAL TRAFFIC PREDICTION")
    print("=" * 70)
    print("Training samples:", len(train))
    print("Testing samples:", len(test))
    print("Training seeds: 1, 2")
    print("Testing seed: 3")
    print("Input features:", len(FEATURES))
    print("Prediction targets:", len(DIRECTIONS))

    models = {}
    results = []

    for direction in DIRECTIONS:

        target = f"target_{direction}_queue"

        preprocessing = ColumnTransformer([
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ])

        estimator = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

        model = Pipeline([
            ("preprocessing", preprocessing),
            ("regressor", estimator),
        ])

        print(f"\nTraining {direction.upper()} XGBoost model...")
        model.fit(train[FEATURES], train[target])

        models[direction] = model

        for scenario in ["low", "medium", "high"]:

            group = test[test["scenario"] == scenario]

            actual = group[target].to_numpy(dtype=float)
            prediction = np.maximum(
                model.predict(group[FEATURES]), 0
            )

            zero = np.zeros(len(actual))

            persistence = group[
                f"observed_{direction}_queue"
            ].to_numpy(dtype=float)

            for name, predicted in [
                ("Zero", zero),
                ("Persistence", persistence),
                ("XGBoost", prediction),
            ]:

                values = calculate_metrics(actual, predicted)

                results.append({
                    "Scenario": scenario,
                    "Direction": direction,
                    "Model": name,
                    "Samples": len(actual),
                    "NonzeroTargets": int((actual > 0).sum()),
                    **{
                        metric: round(float(value), 6)
                        for metric, value in values.items()
                    },
                })

            xgb_metrics = calculate_metrics(actual, prediction)

            print(
                f"{scenario.upper():<7} "
                f"MAE={xgb_metrics['MAE']:.4f} | "
                f"RMSE={xgb_metrics['RMSE']:.4f} | "
                f"Nonzero MAE={xgb_metrics['NonzeroMAE']:.4f} | "
                f"F1={xgb_metrics['F1']:.4f}"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(results).to_csv(
        METRICS_FILE,
        index=False,
    )

    joblib.dump({
        "models": models,
        "features": FEATURES,
        "targets": targets,
        "training_seeds": [1, 2],
        "test_seed": 3,
        "prediction_horizon_seconds": 10,
        "prediction_threshold": 0.5,
    }, MODEL_FILE)

    print("\n" + "=" * 70)
    print("XGBOOST TRAINING COMPLETE")
    print("Model:", MODEL_FILE)
    print("Metrics:", METRICS_FILE)
    print("=" * 70)


if __name__ == "__main__":
    main()
