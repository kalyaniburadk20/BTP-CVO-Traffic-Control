"""Seed-rotated robustness evaluation for the BTP ML extensions.

This script evaluates two existing pipelines without changing their current
models, metrics, datasets, or the original SUMO experiment outputs:

1. Temporal Random Forest traffic prediction:
   - 3 folds; each fold trains on two seeds and tests on the held-out seed.
   - Compares against zero-queue and persistence baselines.

2. Fresh-queue learned-CVO XGBoost:
   - 3 rotations; one seed for training, one for validation/calibration,
     and one untouched seed for testing.
   - Fits isotonic calibration and selects the F1/precision threshold using
     validation data only.

Outputs are written to results/processed/ml/seed_robustness/.
"""
from __future__ import annotations

import json

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = PROJECT_ROOT / "results" / "processed" / "ml"
OUTPUT_DIR = ML_DIR / "seed_robustness"

TEMPORAL_DATASET = ML_DIR / "traffic_prediction_dataset_temporal_v2.csv"
FRESH_CVO_DATASET = ML_DIR / "learned_cvo_fresh_queue_dataset.csv"

SEEDS = (1, 2, 3)
DIRECTIONS = ("north", "south", "east", "west")
SCENARIOS = ("low", "medium", "high")

TEMPORAL_NUMERIC_FEATURES = [
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
TEMPORAL_CATEGORICAL_FEATURES = [
    "scenario", "strategy", "intersection_id"
]
TEMPORAL_FEATURES = (
    TEMPORAL_NUMERIC_FEATURES + TEMPORAL_CATEGORICAL_FEATURES
)
TEMPORAL_TARGETS = [f"target_{direction}_queue" for direction in DIRECTIONS]

CVO_NUMERIC_FEATURES = [
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
CVO_CATEGORICAL_FEATURES = [
    "current_current_phase", "scenario", "intersection_id"
]
CVO_FEATURES = CVO_NUMERIC_FEATURES + CVO_CATEGORICAL_FEATURES
CVO_TARGET = "fresh_cvo"

TEMPORAL_DETAIL = OUTPUT_DIR / "traffic_prediction_temporal_seed_cv.csv"
TEMPORAL_SUMMARY = OUTPUT_DIR / "traffic_prediction_temporal_seed_cv_summary.csv"
CVO_DETAIL = OUTPUT_DIR / "learned_cvo_xgboost_isotonic_seed_cv.csv"
CVO_SUMMARY = OUTPUT_DIR / "learned_cvo_xgboost_isotonic_seed_cv_summary.csv"
AUDIT_JSON = OUTPUT_DIR / "seed_robustness_run_manifest.json"


def require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def check_seed_coverage(df: pd.DataFrame, name: str) -> None:
    found = set(pd.to_numeric(df["seed"], errors="raise").astype(int).unique())
    if found != set(SEEDS):
        raise ValueError(f"{name}: expected seeds {SEEDS}, found {sorted(found)}")


def temporal_regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(np.asarray(predicted, dtype=float), 0.0)
    positive = actual > 0
    predicted_positive = predicted >= 0.5
    nonzero_mae = (
        mean_absolute_error(actual[positive], predicted[positive])
        if positive.any() else float("nan")
    )
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "NonzeroMAE": float(nonzero_mae),
        "Precision": float(precision_score(positive, predicted_positive, zero_division=0)),
        "Recall": float(recall_score(positive, predicted_positive, zero_division=0)),
        "F1": float(f1_score(positive, predicted_positive, zero_division=0)),
    }


def summarize_folds(
    details: pd.DataFrame,
    group_columns: list[str],
    metric_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in details.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["folds"] = int(group["test_seed"].nunique())
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def load_and_validate_temporal() -> pd.DataFrame:
    if not TEMPORAL_DATASET.exists():
        raise FileNotFoundError(TEMPORAL_DATASET)
    df = pd.read_csv(TEMPORAL_DATASET)
    required = TEMPORAL_FEATURES + TEMPORAL_TARGETS + [
        "run_id", "seed", "time", "target_time", "last_observation_time",
        "observation_age", "previous_observation_interval",
    ]
    require_columns(df, required, "Temporal dataset")
    check_seed_coverage(df, "Temporal dataset")

    numeric_columns = TEMPORAL_NUMERIC_FEATURES + TEMPORAL_TARGETS + [
        "seed", "time", "target_time", "last_observation_time",
        "observation_age", "previous_observation_interval",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")

    model_columns = TEMPORAL_FEATURES + TEMPORAL_TARGETS
    if df[model_columns].isna().any().any():
        raise ValueError("Temporal dataset contains missing model features or targets.")
    if (df[TEMPORAL_TARGETS] < 0).any().any():
        raise ValueError("Temporal dataset contains negative queue targets.")
    if df.duplicated(["run_id", "intersection_id", "time"]).any():
        raise ValueError("Temporal dataset has duplicate run/intersection/time rows.")
    if not np.allclose(df["target_time"] - df["time"], 10.0, atol=1e-6):
        raise ValueError("Temporal target horizon is not exactly 10 seconds for every row.")
    if (df["last_observation_time"] > df["time"] + 1e-8).any():
        raise ValueError("Temporal dataset has an observation timestamp later than its sample time.")
    if not np.allclose(
        df["observation_age"],
        df["time"] - df["last_observation_time"],
        atol=1e-6,
    ):
        raise ValueError("observation_age does not match time - last_observation_time.")
    run_seed_counts = df.groupby("run_id")["seed"].nunique()
    if (run_seed_counts != 1).any():
        raise ValueError("A temporal run_id maps to multiple seeds.")
    return df


def run_temporal_seed_cv(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for test_seed in SEEDS:
        train = df[df["seed"] != test_seed].copy()
        test = df[df["seed"] == test_seed].copy()
        train_runs = set(train["run_id"].unique())
        test_runs = set(test["run_id"].unique())
        if not len(train) or not len(test) or train_runs.intersection(test_runs):
            raise ValueError(f"Invalid temporal split for held-out seed {test_seed}.")

        preprocessor = ColumnTransformer([
            ("numeric", "passthrough", TEMPORAL_NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), TEMPORAL_CATEGORICAL_FEATURES),
        ])
        model = Pipeline([
            ("preprocessing", preprocessor),
            ("regressor", RandomForestRegressor(
                n_estimators=100,
                max_depth=16,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            )),
        ])
        model.fit(train[TEMPORAL_FEATURES], train[TEMPORAL_TARGETS])
        model_predictions = np.maximum(model.predict(test[TEMPORAL_FEATURES]), 0.0)

        for scenario in SCENARIOS:
            scenario_mask = test["scenario"].eq(scenario).to_numpy()
            scenario_test = test.loc[scenario_mask]
            if scenario_test.empty:
                raise ValueError(f"No {scenario} samples in test seed {test_seed}.")

            for target_index, direction in enumerate(DIRECTIONS):
                actual = scenario_test[f"target_{direction}_queue"].to_numpy(dtype=float)
                predictions = {
                    "Zero": np.zeros(len(actual), dtype=float),
                    "Persistence": scenario_test[f"observed_{direction}_queue"].to_numpy(dtype=float),
                    "TemporalRF": model_predictions[scenario_mask, target_index],
                }
                for model_name, predicted in predictions.items():
                    row: dict[str, Any] = {
                        "test_seed": test_seed,
                        "train_seeds": ";".join(str(seed) for seed in SEEDS if seed != test_seed),
                        "scenario": scenario,
                        "direction": direction,
                        "model": model_name,
                        "samples": len(actual),
                        "nonzero_targets": int((actual > 0).sum()),
                    }
                    row.update(temporal_regression_metrics(actual, predicted))
                    rows.append(row)
        print(f"Temporal RF fold complete: held-out seed {test_seed}")
    return pd.DataFrame(rows)


def select_f1_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float, float]:
    best_threshold = 0.5
    best_f1 = -1.0
    best_precision = -1.0
    for threshold in np.arange(0.01, 1.00, 0.01):
        predicted = (probabilities >= threshold).astype(int)
        current_f1 = f1_score(y_true, predicted, zero_division=0)
        current_precision = precision_score(y_true, predicted, zero_division=0)
        if (current_f1, current_precision) > (best_f1, best_precision):
            best_threshold = float(threshold)
            best_f1 = float(current_f1)
            best_precision = float(current_precision)
    return best_threshold, best_f1, best_precision


def cvo_classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    model_name: str,
    test_seed: int,
    train_seed: int | None,
    validation_seed: int | None,
) -> dict[str, Any]:
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1.0 - 1e-7)
    predictions = (probabilities >= threshold).astype(int)
    return {
        "test_seed": test_seed,
        "train_seed": train_seed,
        "validation_seed": validation_seed,
        "model": model_name,
        "threshold": float(threshold),
        "samples": int(len(y_true)),
        "positive_samples": int(np.sum(y_true == 1)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
    }


def load_and_validate_fresh_cvo() -> pd.DataFrame:
    if not FRESH_CVO_DATASET.exists():
        raise FileNotFoundError(FRESH_CVO_DATASET)
    df = pd.read_csv(FRESH_CVO_DATASET)
    require_columns(df, CVO_FEATURES + [CVO_TARGET, "seed", "run_id", "time"], "Fresh-CVO dataset")
    check_seed_coverage(df, "Fresh-CVO dataset")
    df["seed"] = pd.to_numeric(df["seed"], errors="raise").astype(int)
    df[CVO_TARGET] = pd.to_numeric(df[CVO_TARGET], errors="raise").astype(int)
    if df[CVO_FEATURES + [CVO_TARGET]].isna().any().any():
        raise ValueError("Fresh-CVO dataset contains missing model features or labels.")
    if not set(df[CVO_TARGET].unique()).issubset({0, 1}):
        raise ValueError("Fresh-CVO target must be binary.")
    if df.duplicated(["run_id", "intersection_id", "time"]).any():
        raise ValueError("Fresh-CVO dataset has duplicate run/intersection/time rows.")
    for seed in SEEDS:
        labels = set(df.loc[df["seed"] == seed, CVO_TARGET].unique())
        if labels != {0, 1}:
            raise ValueError(f"Fresh-CVO seed {seed} must contain both classes; found {labels}.")
    run_seed_counts = df.groupby("run_id")["seed"].nunique()
    if (run_seed_counts != 1).any():
        raise ValueError("A fresh-CVO run_id maps to multiple seeds.")
    return df


def run_fresh_cvo_seed_cv(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    # Rotate roles so every seed is tested once. The test seed is never used
    # for fitting, calibration, or threshold selection in its fold.
    rotations = ((1, 2, 3), (2, 3, 1), (3, 1, 2))  # train, validation, test
    for train_seed, validation_seed, test_seed in rotations:
        train = df[df["seed"] == train_seed].copy()
        validation = df[df["seed"] == validation_seed].copy()
        test = df[df["seed"] == test_seed].copy()

        preprocessor = ColumnTransformer([
            ("numeric", "passthrough", CVO_NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CVO_CATEGORICAL_FEATURES),
        ])
        classifier = XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            min_child_weight=5,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=3.0,
            reg_alpha=0.1,
            scale_pos_weight=float((train[CVO_TARGET] == 0).sum() / (train[CVO_TARGET] == 1).sum()),
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )
        model = Pipeline([
            ("preprocessing", preprocessor),
            ("classifier", classifier),
        ])
        model.fit(train[CVO_FEATURES], train[CVO_TARGET])

        y_validation = validation[CVO_TARGET].to_numpy(dtype=int)
        y_test = test[CVO_TARGET].to_numpy(dtype=int)
        raw_validation = model.predict_proba(validation[CVO_FEATURES])[:, 1]
        raw_test = model.predict_proba(test[CVO_FEATURES])[:, 1]

        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_validation, y_validation)
        calibrated_validation = np.clip(calibrator.predict(raw_validation), 1e-7, 1.0 - 1e-7)
        calibrated_test = np.clip(calibrator.predict(raw_test), 1e-7, 1.0 - 1e-7)
        selected_threshold, validation_f1, validation_precision = select_f1_threshold(
            y_validation, calibrated_validation
        )

        rows.append(cvo_classification_metrics(
            y_test, raw_test, 0.5, "XGBoost_raw_threshold_0.50",
            test_seed, train_seed, validation_seed,
        ))
        calibrated_row = cvo_classification_metrics(
            y_test, calibrated_test, selected_threshold,
            "XGBoost_isotonic_validation_threshold",
            test_seed, train_seed, validation_seed,
        )
        calibrated_row["validation_f1"] = validation_f1
        calibrated_row["validation_precision"] = validation_precision
        rows.append(calibrated_row)

        baseline_probabilities = np.zeros(len(y_test), dtype=float)
        baseline_predictions = np.zeros(len(y_test), dtype=int)
        rows.append({
            "test_seed": test_seed,
            "train_seed": train_seed,
            "validation_seed": validation_seed,
            "model": "Always_predict_zero",
            "threshold": 0.5,
            "samples": int(len(y_test)),
            "positive_samples": int(np.sum(y_test == 1)),
            "accuracy": float(accuracy_score(y_test, baseline_predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, baseline_predictions)),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": float("nan"),
            "average_precision": float("nan"),
            "brier_score": float(brier_score_loss(y_test, baseline_probabilities)),
            "log_loss": float("nan"),
        })
        print(
            "Fresh-CVO XGBoost fold complete: "
            f"train={train_seed}, validation={validation_seed}, test={test_seed}; "
            f"calibrated threshold={selected_threshold:.2f}, "
            f"validation F1={validation_f1:.4f}"
        )
    return pd.DataFrame(rows)


def main() -> None:
    output_paths = [TEMPORAL_DETAIL, TEMPORAL_SUMMARY, CVO_DETAIL, CVO_SUMMARY, AUDIT_JSON]
    existing = [path for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing seed-robustness outputs: "
            + ", ".join(str(path) for path in existing)
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading and validating temporal traffic dataset...")
    temporal_df = load_and_validate_temporal()
    print(
        f"Temporal data: {len(temporal_df):,} rows, "
        f"{temporal_df['run_id'].nunique()} runs, "
        f"{temporal_df['intersection_id'].nunique()} intersections"
    )
    temporal_details = run_temporal_seed_cv(temporal_df)
    temporal_summary = summarize_folds(
        temporal_details,
        ["scenario", "direction", "model"],
        ["MAE", "RMSE", "NonzeroMAE", "Precision", "Recall", "F1"],
    )
    temporal_details.to_csv(TEMPORAL_DETAIL, index=False)
    temporal_summary.to_csv(TEMPORAL_SUMMARY, index=False)

    print("\nLoading and validating fresh-queue CVO dataset...")
    cvo_df = load_and_validate_fresh_cvo()
    print(
        f"Fresh-CVO data: {len(cvo_df):,} rows, "
        f"{cvo_df['run_id'].nunique()} runs, "
        f"positive rate={(cvo_df[CVO_TARGET].mean() * 100):.2f}%"
    )
    cvo_details = run_fresh_cvo_seed_cv(cvo_df)
    cvo_summary = summarize_folds(
        cvo_details,
        ["model"],
        ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "average_precision", "brier_score", "log_loss"],
    )
    cvo_details.to_csv(CVO_DETAIL, index=False)
    cvo_summary.to_csv(CVO_SUMMARY, index=False)

    manifest = {
        "evaluation": "three-fold seed-rotated robustness evaluation",
        "seeds": list(SEEDS),
        "temporal_traffic_prediction": {
            "dataset": str(TEMPORAL_DATASET.relative_to(PROJECT_ROOT)),
            "prediction_horizon_seconds": 10,
            "training_rule": "train on the two non-held-out seeds; test on the held-out seed",
            "model": "RandomForestRegressor",
            "baselines": ["zero queue", "persistence using latest observed queue"],
            "folds": 3,
        },
        "fresh_queue_learned_cvo": {
            "dataset": str(FRESH_CVO_DATASET.relative_to(PROJECT_ROOT)),
            "training_rule": "one training seed, one validation/calibration seed, one untouched test seed; rotate all roles",
            "model": "XGBClassifier with isotonic probability calibration",
            "threshold_rule": "select F1, then precision, using validation data only",
            "folds": 3,
        },
        "note": "This is an offline evaluation on existing datasets; it does not rerun SUMO or overwrite existing model artifacts/results.",
    }
    AUDIT_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nTEMPORAL RF CROSS-SEED SUMMARY")
    print(temporal_summary.to_string(index=False))
    print("\nFRESH-CVO XGBOOST CROSS-SEED SUMMARY")
    print(cvo_summary.to_string(index=False))
    print("\nSaved new evaluation outputs under:", OUTPUT_DIR)
    for path in output_paths:
        print(" -", path.name)


if __name__ == "__main__":
    main()
