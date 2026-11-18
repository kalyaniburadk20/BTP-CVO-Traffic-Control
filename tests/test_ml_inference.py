"""Smoke and contract tests for the BTP ML inference interface.

Run from the repository root:
    python tests/test_ml_inference.py

These tests load existing trained artifacts and use held-out examples. They do
not fit models, alter datasets, or overwrite model artifacts.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.inference import (  # noqa: E402
    CVO_MODEL_PATH,
    ML_RESULTS_DIR,
    TRAFFIC_MODEL_PATH,
    load_fresh_cvo_model,
    load_traffic_model,
    predict_fresh_cvo,
    predict_traffic_queues,
)


class TestMLInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        required_paths = [
            TRAFFIC_MODEL_PATH,
            CVO_MODEL_PATH,
            ML_RESULTS_DIR / "traffic_prediction_dataset_temporal_v2.csv",
            ML_RESULTS_DIR / "learned_cvo_fresh_queue_dataset.csv",
        ]
        missing = [str(path) for path in required_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Required trained artifacts or demo datasets are missing:\n"
                + "\n".join(missing)
            )

        cls.traffic_artifact = load_traffic_model()
        cls.cvo_artifact = load_fresh_cvo_model()

        traffic_data = pd.read_csv(required_paths[2])
        cvo_data = pd.read_csv(required_paths[3])
        cls.traffic_seed3 = traffic_data.loc[
            traffic_data["seed"] == 3
        ].head(3).copy()
        cls.cvo_seed3 = cvo_data.loc[
            cvo_data["seed"] == 3
        ].head(3).copy()

        if cls.traffic_seed3.empty or cls.cvo_seed3.empty:
            raise ValueError("Held-out seed-3 examples are required for inference tests.")

    def test_traffic_prediction_returns_four_finite_nonnegative_queues(self) -> None:
        features = self.traffic_seed3[self.traffic_artifact["features"]]
        predictions = predict_traffic_queues(features, self.traffic_artifact)

        expected_columns = [
            "predicted_north_queue",
            "predicted_south_queue",
            "predicted_east_queue",
            "predicted_west_queue",
        ]
        self.assertEqual(list(predictions.columns), expected_columns)
        self.assertEqual(predictions.shape, (len(features), 4))
        self.assertTrue(np.isfinite(predictions.to_numpy()).all())
        self.assertTrue((predictions.to_numpy() >= 0).all())

    def test_cvo_prediction_returns_valid_probability_and_class(self) -> None:
        features = self.cvo_seed3[self.cvo_artifact["features"]]
        predictions = predict_fresh_cvo(features, self.cvo_artifact)

        self.assertEqual(len(predictions), len(features))
        self.assertTrue(
            predictions["fresh_cvo_probability"].between(0.0, 1.0).all()
        )
        threshold = float(self.cvo_artifact["threshold"])
        self.assertTrue(
            np.allclose(predictions["classification_threshold"], threshold)
        )
        expected_class = (
            predictions["fresh_cvo_probability"].to_numpy() >= threshold
        ).astype(int)
        np.testing.assert_array_equal(
            predictions["predicted_fresh_cvo"].to_numpy(), expected_class
        )

    def test_missing_traffic_feature_is_rejected(self) -> None:
        row = self.traffic_seed3.iloc[0][self.traffic_artifact["features"]].to_dict()
        row.pop(self.traffic_artifact["features"][0])
        with self.assertRaisesRegex(ValueError, "missing required features"):
            predict_traffic_queues(row, self.traffic_artifact)

    def test_missing_cvo_feature_is_rejected(self) -> None:
        row = self.cvo_seed3.iloc[0][self.cvo_artifact["features"]].to_dict()
        row.pop(self.cvo_artifact["features"][0])
        with self.assertRaisesRegex(ValueError, "missing required features"):
            predict_fresh_cvo(row, self.cvo_artifact)

    def test_missing_values_are_rejected(self) -> None:
        row = self.traffic_seed3.iloc[0][self.traffic_artifact["features"]].to_dict()
        row[self.traffic_artifact["features"][0]] = np.nan
        with self.assertRaisesRegex(ValueError, "missing values"):
            predict_traffic_queues(row, self.traffic_artifact)

    def test_empty_batch_is_rejected(self) -> None:
        empty = pd.DataFrame(columns=self.traffic_artifact["features"])
        with self.assertRaisesRegex(ValueError, "contains no rows"):
            predict_traffic_queues(empty, self.traffic_artifact)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestMLInference)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
