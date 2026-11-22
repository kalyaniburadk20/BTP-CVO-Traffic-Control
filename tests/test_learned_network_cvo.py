import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.traffic_state import TrafficState
from src.perception.belief import Belief
from src.ml.inference import load_fresh_cvo_model
from src.ml.network_cvo import (
    build_fresh_cvo_features,
    predict_network_fresh_cvo,
)


def make_belief(
    north=12,
    south=8,
    east=5,
    west=3,
    phase=0,
    vehicle_count=25,
    observed_at=10.0,
):
    state = TrafficState(
        north_queue=north,
        south_queue=south,
        east_queue=east,
        west_queue=west,
        north_waiting=12.5,
        south_waiting=8.5,
        east_waiting=4.0,
        west_waiting=3.0,
        current_phase=phase,
        vehicle_count=vehicle_count,
    )
    return Belief(state=state, last_observation_time=observed_at)


class TestLearnedNetworkCVO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = load_fresh_cvo_model()

    def test_feature_adapter_matches_training_schema(self):
        belief = make_belief()
        row = build_fresh_cvo_features(
            belief=belief,
            current_time=14.0,
            scenario="low",
            intersection_id="J1",
        )

        required = set(self.artifact["features"])
        self.assertEqual(set(row), required)
        self.assertEqual(row["current_north_queue"], 12)
        self.assertEqual(row["current_south_queue"], 8)
        self.assertEqual(row["current_east_queue"], 5)
        self.assertEqual(row["current_west_queue"], 3)
        self.assertEqual(row["current_current_phase"], 0)
        self.assertEqual(row["current_vehicle_count"], 25)
        self.assertEqual(row["belief_age"], 4.0)
        self.assertEqual(row["decision_margin"], 12)
        self.assertEqual(row["scenario"], "low")
        self.assertEqual(row["intersection_id"], "J1")

    def test_prediction_returns_one_probability_per_intersection(self):
        beliefs = {
            "J1": make_belief(north=20, east=5),
            "J2": make_belief(north=5, east=20, phase=2, observed_at=12.0),
            "J3": make_belief(north=10, east=10, phase=0),
        }
        probabilities = predict_network_fresh_cvo(
            beliefs=beliefs,
            current_time=15.0,
            scenario="low",
            artifact=self.artifact,
        )

        self.assertEqual(set(probabilities), set(beliefs))
        for probability in probabilities.values():
            self.assertGreaterEqual(probability, 0.0)
            self.assertLessEqual(probability, 1.0)

    def test_invalid_scenario_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "scenario must be"):
            build_fresh_cvo_features(
                make_belief(),
                current_time=11.0,
                scenario="unexpected",
                intersection_id="J1",
            )

    def test_future_belief_timestamp_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot precede"):
            build_fresh_cvo_features(
                make_belief(observed_at=20.0),
                current_time=10.0,
                scenario="low",
                intersection_id="J1",
            )

    def test_missing_belief_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "belief containing"):
            build_fresh_cvo_features(
                None,
                current_time=10.0,
                scenario="low",
                intersection_id="J1",
            )

    def test_empty_network_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "At least one"):
            predict_network_fresh_cvo(
                beliefs={},
                current_time=10.0,
                scenario="low",
                artifact=self.artifact,
            )


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestLearnedNetworkCVO
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
