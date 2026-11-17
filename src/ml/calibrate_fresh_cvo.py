from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


DATA_PATH = Path(
    "results/processed/ml/learned_cvo_fresh_queue_dataset.csv"
)
OUTPUT_DIR = Path("results/processed/ml")

MODEL_PATHS = {
    "random_forest": OUTPUT_DIR
    / "learned_cvo_fresh_queue_random_forest.joblib",
    "xgboost": OUTPUT_DIR
    / "learned_cvo_fresh_queue_xgboost.joblib",
}

METRICS_PATH = (
    OUTPUT_DIR / "learned_cvo_fresh_queue_calibration_metrics.csv"
)

METHODS = ["sigmoid", "isotonic"]


def calculate_metrics(y_true, probabilities, threshold):
    probabilities = np.clip(
        np.asarray(probabilities, dtype=float),
        1e-7,
        1 - 1e-7,
    )
    predictions = (probabilities >= threshold).astype(int)

    return {
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
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(
            y_true, probabilities
        ),
        "brier_score": brier_score_loss(
            y_true, probabilities
        ),
        "log_loss": log_loss(
            y_true, probabilities, labels=[0, 1]
        ),
    }


def select_threshold(y_true, probabilities):
    best = None

    for threshold in np.arange(0.01, 1.00, 0.01):
        predictions = (
            probabilities >= threshold
        ).astype(int)

        result = {
            "threshold": float(threshold),
            "f1": f1_score(
                y_true, predictions, zero_division=0
            ),
            "precision": precision_score(
                y_true, predictions, zero_division=0
            ),
        }

        if best is None or (
            result["f1"], result["precision"]
        ) > (
            best["f1"], best["precision"]
        ):
            best = result

    return best


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    if METRICS_PATH.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {METRICS_PATH}"
        )

    output_paths = [
        OUTPUT_DIR
        / f"learned_cvo_fresh_queue_{model_name}_{method}_calibrated.joblib"
        for model_name in MODEL_PATHS
        for method in METHODS
    ]

    existing = [path for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    df = pd.read_csv(DATA_PATH)
    train = df[df["seed"] == 1]
    validation = df[df["seed"] == 2]
    test = df[df["seed"] == 3]

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("A seed split is empty.")

    if set(validation["fresh_cvo"].unique()) != {0, 1}:
        raise ValueError("Validation split must contain both classes.")

    if set(test["fresh_cvo"].unique()) != {0, 1}:
        raise ValueError("Test split must contain both classes.")

    y_val = validation["fresh_cvo"].astype(int)
    y_test = test["fresh_cvo"].astype(int)

    rows = []

    for model_name, model_path in MODEL_PATHS.items():
        if not model_path.exists():
            raise FileNotFoundError(model_path)

        artifact = joblib.load(model_path)
        base_model = artifact["model"]
        features = artifact["features"]

        X_val = validation[features]
        X_test = test[features]

        raw_val = base_model.predict_proba(X_val)[:, 1]
        raw_test = base_model.predict_proba(X_test)[:, 1]

        # Record the previously trained, uncalibrated model.
        raw_threshold = float(artifact["threshold"])
        raw_metrics = calculate_metrics(
            y_test, raw_test, raw_threshold
        )
        raw_metrics.update({
            "model": model_name,
            "calibration": "none",
            "validation_brier": brier_score_loss(
                y_val, raw_val
            ),
            "validation_log_loss": log_loss(
                y_val, np.clip(raw_val, 1e-7, 1 - 1e-7),
                labels=[0, 1],
            ),
        })
        rows.append(raw_metrics)

        for method in METHODS:
            if method == "sigmoid":
                calibrator = LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                )
                calibrator.fit(
                    raw_val.reshape(-1, 1),
                    y_val,
                )

                calibrated_val = calibrator.predict_proba(
                    raw_val.reshape(-1, 1)
                )[:, 1]

                calibrated_test = calibrator.predict_proba(
                    raw_test.reshape(-1, 1)
                )[:, 1]

            else:
                calibrator = IsotonicRegression(
                    out_of_bounds="clip"
                )
                calibrator.fit(raw_val, y_val)

                calibrated_val = calibrator.predict(raw_val)
                calibrated_test = calibrator.predict(raw_test)

            calibrated_val = np.clip(
                calibrated_val, 1e-7, 1 - 1e-7
            )
            calibrated_test = np.clip(
                calibrated_test, 1e-7, 1 - 1e-7
            )

            selected = select_threshold(
                y_val, calibrated_val
            )

            result = calculate_metrics(
                y_test,
                calibrated_test,
                selected["threshold"],
            )

            result.update({
                "model": model_name,
                "calibration": method,
                "validation_brier": brier_score_loss(
                    y_val, calibrated_val
                ),
                "validation_log_loss": log_loss(
                    y_val,
                    calibrated_val,
                    labels=[0, 1],
                ),
                "validation_f1": selected["f1"],
                "validation_precision": selected["precision"],
            })

            rows.append(result)

            calibrated_path = (
                OUTPUT_DIR
                / f"learned_cvo_fresh_queue_{model_name}_{method}_calibrated.joblib"
            )

            joblib.dump(
                {
                    "base_model": base_model,
                    "calibrator": calibrator,
                    "calibration_method": method,
                    "threshold": selected["threshold"],
                    "features": features,
                    "target": "fresh_cvo",
                    "train_seed": 1,
                    "calibration_seed": 2,
                    "test_seed": 3,
                },
                calibrated_path,
            )

    # Always-negative probability reference.
    baseline_probabilities = np.zeros(len(y_test))
    baseline_row = {
        "model": "always_predict_zero",
        "calibration": "none",
        "threshold": 0.50,
        "accuracy": accuracy_score(
            y_test, np.zeros(len(y_test), dtype=int)
        ),
        "balanced_accuracy": 0.50,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "roc_auc": np.nan,
        "average_precision": np.nan,
        "brier_score": brier_score_loss(
            y_test, baseline_probabilities
        ),
        "log_loss": np.nan,
        "validation_brier": np.nan,
        "validation_log_loss": np.nan,
    }
    rows.append(baseline_row)

    result_df = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(METRICS_PATH, index=False)

    print("=" * 70)
    print("FRESH-QUEUE CVO PROBABILITY CALIBRATION COMPLETE")
    print("=" * 70)
    print("Calibration data: Seed 2")
    print("Final evaluation: Seed 3")
    print("\nResults:")
    print(result_df.to_string(index=False))

    print("\nSaved metrics:")
    print(METRICS_PATH)

    print("\nCalibration artifacts:")
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
