"""Reusable inference helpers for the BTP traffic ML models.

This module loads the already-trained artifacts and exposes two prediction
functions:

* predict_traffic_queues: 10-second, four-direction queue estimates from the
  temporal Random Forest pipeline.
* predict_fresh_cvo: calibrated probability that fresh queue measurements
  would change the controller's action under the dataset's label definition.

The learned-CVO output is a classification estimate, not a complete
observation-acquisition decision. Acquisition cost and the network observation
budget remain separate parts of the controller.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_RESULTS_DIR = PROJECT_ROOT / "results" / "processed" / "ml"

TRAFFIC_MODEL_PATH = ML_RESULTS_DIR / "traffic_prediction_temporal_rf.joblib"
CVO_MODEL_PATH = (
    ML_RESULTS_DIR / "learned_cvo_fresh_queue_xgboost_isotonic_calibrated.joblib"
)


def _to_frame(
    features: pd.DataFrame | Mapping[str, Any] | list[Mapping[str, Any]],
    required_columns: list[str],
    model_name: str,
) -> pd.DataFrame:
    """Normalize input and validate required model features."""
    if isinstance(features, pd.DataFrame):
        frame = features.copy()
    elif isinstance(features, Mapping):
        frame = pd.DataFrame([dict(features)])
    elif isinstance(features, list) and all(
        isinstance(item, Mapping) for item in features
    ):
        frame = pd.DataFrame([dict(item) for item in features])
    else:
        raise TypeError(
            f"{model_name} input must be a DataFrame, mapping, "
            "or list of mappings."
        )

    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{model_name} input is missing required features: {missing}"
        )

    frame = frame.loc[:, required_columns]
    if frame.empty:
        raise ValueError(f"{model_name} input contains no rows.")
    if frame.isna().any().any():
        bad_columns = frame.columns[frame.isna().any()].tolist()
        raise ValueError(
            f"{model_name} input contains missing values in: {bad_columns}"
        )
    return frame


def load_traffic_model(model_path: Path | str = TRAFFIC_MODEL_PATH) -> dict[str, Any]:
    """Load and validate the temporal traffic prediction artifact."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Temporal traffic model not found: {path}")

    artifact = joblib.load(path)
    required_keys = {"model", "features", "targets"}
    missing = required_keys - set(artifact)
    if missing:
        raise ValueError(f"Traffic model artifact missing keys: {sorted(missing)}")

    if len(artifact["targets"]) != 4:
        raise ValueError(
            "Expected four directional queue targets in temporal model artifact."
        )
    return artifact


def load_fresh_cvo_model(model_path: Path | str = CVO_MODEL_PATH) -> dict[str, Any]:
    """Load and validate the isotonic-calibrated fresh-queue CVO artifact."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Calibrated fresh-CVO model not found: {path}")

    artifact = joblib.load(path)
    required_keys = {
        "base_model",
        "calibrator",
        "calibration_method",
        "threshold",
        "features",
    }
    missing = required_keys - set(artifact)
    if missing:
        raise ValueError(f"Fresh-CVO artifact missing keys: {sorted(missing)}")

    method = str(artifact["calibration_method"]).lower()
    if method not in {"isotonic", "sigmoid"}:
        raise ValueError(f"Unsupported calibration method in artifact: {method}")
    threshold = float(artifact["threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Invalid CVO classification threshold: {threshold}")
    return artifact


def predict_traffic_queues(
    features: pd.DataFrame | Mapping[str, Any] | list[Mapping[str, Any]],
    artifact: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Predict 10-second-ahead queues for north, south, east, and west.

    Returns columns named predicted_north_queue, predicted_south_queue,
    predicted_east_queue, and predicted_west_queue when those targets are
    present in the saved model metadata.
    """
    model_artifact = artifact or load_traffic_model()
    feature_names = list(model_artifact["features"])
    target_names = list(model_artifact["targets"])
    frame = _to_frame(features, feature_names, "Temporal traffic prediction")

    predictions = np.asarray(model_artifact["model"].predict(frame), dtype=float)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    if predictions.shape != (len(frame), len(target_names)):
        raise ValueError(
            "Temporal model returned an unexpected prediction shape: "
            f"{predictions.shape}; expected {(len(frame), len(target_names))}."
        )
    if not np.isfinite(predictions).all():
        raise ValueError("Temporal model returned non-finite predictions.")

    predictions = np.maximum(predictions, 0.0)
    output_names = [
        name.replace("target_", "predicted_") for name in target_names
    ]
    return pd.DataFrame(predictions, columns=output_names, index=frame.index)


def predict_fresh_cvo(
    features: pd.DataFrame | Mapping[str, Any] | list[Mapping[str, Any]],
    artifact: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Estimate calibrated fresh-queue action-divergence probability.

    `predicted_fresh_cvo` is the model's binary classification at the
    validation-selected threshold stored in the artifact. It is not by itself
    an instruction to acquire an observation: the acquisition-cost comparison
    and network-level observation budget are not implemented here.
    """
    model_artifact = artifact or load_fresh_cvo_model()
    feature_names = list(model_artifact["features"])
    frame = _to_frame(features, feature_names, "Fresh-CVO prediction")

    raw_probabilities = np.asarray(
        model_artifact["base_model"].predict_proba(frame)[:, 1], dtype=float
    )
    method = str(model_artifact["calibration_method"]).lower()
    calibrator = model_artifact["calibrator"]

    if method == "isotonic":
        calibrated = np.asarray(calibrator.predict(raw_probabilities), dtype=float)
    elif method == "sigmoid":
        calibrated = np.asarray(
            calibrator.predict_proba(raw_probabilities.reshape(-1, 1))[:, 1],
            dtype=float,
        )
    else:  # Defensive: load_fresh_cvo_model already validates this.
        raise ValueError(f"Unsupported calibration method: {method}")

    calibrated = np.clip(calibrated, 0.0, 1.0)
    if not np.isfinite(calibrated).all():
        raise ValueError("Fresh-CVO model returned non-finite probabilities.")

    threshold = float(model_artifact["threshold"])
    predicted_class = (calibrated >= threshold).astype(int)
    return pd.DataFrame(
        {
            "fresh_cvo_probability": calibrated,
            "classification_threshold": threshold,
            "predicted_fresh_cvo": predicted_class,
        },
        index=frame.index,
    )


def main() -> None:
    """Smoke-test both saved models on one held-out row from each dataset."""
    temporal_path = ML_RESULTS_DIR / "traffic_prediction_dataset_temporal_v2.csv"
    cvo_path = ML_RESULTS_DIR / "learned_cvo_fresh_queue_dataset.csv"
    for path in (temporal_path, cvo_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required inference demo dataset not found: {path}")

    traffic_artifact = load_traffic_model()
    cvo_artifact = load_fresh_cvo_model()

    traffic_data = pd.read_csv(temporal_path)
    traffic_test = traffic_data.loc[traffic_data["seed"] == 3]
    if traffic_test.empty:
        raise ValueError("No held-out seed-3 temporal sample found.")
    traffic_row = traffic_test.iloc[[0]]
    traffic_features = traffic_row[traffic_artifact["features"]]
    traffic_prediction = predict_traffic_queues(
        traffic_features, traffic_artifact
    )

    cvo_data = pd.read_csv(cvo_path)
    cvo_test = cvo_data.loc[
        (cvo_data["seed"] == 3) & (cvo_data["fresh_cvo"] == 1)
    ]
    if cvo_test.empty:
        cvo_test = cvo_data.loc[cvo_data["seed"] == 3]
    if cvo_test.empty:
        raise ValueError("No held-out seed-3 fresh-CVO sample found.")
    cvo_row = cvo_test.iloc[[0]]
    cvo_features = cvo_row[cvo_artifact["features"]]
    cvo_prediction = predict_fresh_cvo(cvo_features, cvo_artifact)

    print("=" * 72)
    print("BTP MACHINE-LEARNING INFERENCE SMOKE TEST")
    print("=" * 72)
    print("Temporal traffic model:")
    print(
        f"  Training seeds: {traffic_artifact.get('training_seeds', 'metadata unavailable')}"
    )
    print(
        f"  Prediction horizon: {traffic_artifact.get('prediction_horizon_seconds', 'metadata unavailable')} seconds"
    )
    print("  Held-out seed-3 prediction:")
    print(traffic_prediction.to_string(index=False))
    actual_queue = {
        column.replace("target_", "actual_"): float(traffic_row.iloc[0][column])
        for column in traffic_artifact["targets"]
    }
    print("  Corresponding target values:")
    print(pd.DataFrame([actual_queue]).to_string(index=False))

    print("\nCalibrated fresh-queue CVO model:")
    print(f"  Calibration: {cvo_artifact['calibration_method']}")
    print(f"  Stored classification threshold: {float(cvo_artifact['threshold']):.3f}")
    print(f"  Held-out seed-3 actual fresh_cvo label: {int(cvo_row.iloc[0]['fresh_cvo'])}")
    print("  Prediction:")
    print(cvo_prediction.to_string(index=False))

    print("\nSmoke test completed. No model was retrained and no files were overwritten.")
    print(
        "Note: the CVO output estimates action divergence; it is not a standalone "
        "acquisition command."
    )


if __name__ == "__main__":
    main()
