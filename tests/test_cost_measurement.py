import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.acquisition.cost_measurement import CostMeasurementRecorder


class TestCostMeasurementRecorder(unittest.TestCase):
    def test_model_inference_is_timed_and_result_is_preserved(self):
        recorder = CostMeasurementRecorder()
        expected = {"J1": 0.25, "J2": 0.5}

        result = recorder.measure_model_inference(
            simulation_time=12.0,
            operation=lambda: expected,
        )

        self.assertEqual(result, expected)
        self.assertEqual(len(recorder.records), 1)
        row = recorder.records[0]
        self.assertEqual(row["metric"], "model_inference_batch_ms")
        self.assertEqual(row["intersection_id"], "NETWORK")
        self.assertGreaterEqual(row["duration_ms"], 0.0)

    def test_observation_read_and_payload_are_measured_separately(self):
        recorder = CostMeasurementRecorder()
        state_dict = {
            "north_queue": 2,
            "south_queue": 1,
            "east_queue": 0,
            "west_queue": 3,
            "current_phase": 0,
        }
        state = SimpleNamespace(to_dict=lambda: state_dict)

        result = recorder.measure_observation_read(
            simulation_time=20.0,
            intersection_id="J3",
            operation=lambda: state,
        )

        self.assertIs(result, state)
        self.assertEqual(len(recorder.records), 2)

        read_row, payload_row = recorder.records
        self.assertEqual(read_row["metric"], "observation_read_ms")
        self.assertEqual(read_row["intersection_id"], "J3")
        self.assertGreaterEqual(read_row["duration_ms"], 0.0)

        expected_bytes = len(
            json.dumps(
                state_dict,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        )
        self.assertEqual(
            payload_row["metric"], "payload_serialization_ms"
        )
        self.assertEqual(payload_row["payload_bytes"], expected_bytes)
        self.assertGreaterEqual(payload_row["duration_ms"], 0.0)

    def test_summary_contains_runtime_and_payload_statistics(self):
        recorder = CostMeasurementRecorder()
        state = SimpleNamespace(to_dict=lambda: {"queue": 4})

        recorder.measure_model_inference(
            1.0, lambda: {"J1": 0.2}
        )
        recorder.measure_observation_read(
            1.0, "J1", lambda: state
        )

        summary = recorder.summarize()
        self.assertIn("model_inference_batch_ms", summary)
        self.assertIn("observation_read_ms", summary)
        self.assertIn("payload_serialization_ms", summary)
        self.assertIn(
            "mean_duration_ms",
            summary["payload_serialization_ms"],
        )
        self.assertIn(
            "mean_payload_bytes",
            summary["payload_serialization_ms"],
        )

    def test_csv_is_saved_with_expected_columns(self):
        recorder = CostMeasurementRecorder()
        state = SimpleNamespace(to_dict=lambda: {"queue": 1})
        recorder.measure_observation_read(2.0, "J2", lambda: state)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = recorder.save_csv(temp_dir, "low")
            self.assertTrue(output_path.is_file())
            self.assertIn("learned_cvo_cost_low_", output_path.name)

            with output_path.open(
                "r", newline="", encoding="utf-8"
            ) as input_file:
                rows = list(csv.DictReader(input_file))

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            set(rows[0]),
            {
                "simulation_time",
                "intersection_id",
                "metric",
                "duration_ms",
                "payload_bytes",
            },
        )


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestCostMeasurementRecorder
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
