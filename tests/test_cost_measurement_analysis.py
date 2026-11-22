import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_cost_measurements import (
    build_run_metric_rows,
    scenario_summary,
)


class TestCostMeasurementAnalysis(unittest.TestCase):
    def test_bootstrap_and_planner_selected_reads_are_separated(self):
        rows = [
            {
                "simulation_time": "60",
                "intersection_id": "J1",
                "metric": "observation_read_ms",
                "duration_ms": "0.5",
                "payload_bytes": "",
            },
            {
                "simulation_time": "61",
                "intersection_id": "J1",
                "metric": "observation_read_ms",
                "duration_ms": "0.8",
                "payload_bytes": "",
            },
            {
                "simulation_time": "61",
                "intersection_id": "J1",
                "metric": "payload_serialization_ms",
                "duration_ms": "0.02",
                "payload_bytes": "177",
            },
            {
                "simulation_time": "61",
                "intersection_id": "NETWORK",
                "metric": "model_inference_batch_ms",
                "duration_ms": "6.0",
                "payload_bytes": "",
            },
        ]

        summary_rows = build_run_metric_rows(
            path=Path("learned_cvo_cost_low_seed1_example.csv"),
            scenario="low",
            seed=1,
            rows=rows,
        )
        by_metric = {row["metric"]: row for row in summary_rows}

        self.assertEqual(
            by_metric["bootstrap_observation_read_ms"]["sample_count"],
            1,
        )
        self.assertEqual(
            by_metric["planner_selected_observation_read_ms"]["sample_count"],
            1,
        )
        self.assertAlmostEqual(
            by_metric["planner_selected_observation_read_ms"]["mean"],
            0.8,
        )
        self.assertEqual(
            by_metric["planner_selected_payload_bytes"]["mean"],
            177.0,
        )
        self.assertEqual(
            by_metric["model_inference_batch_ms"]["mean"],
            6.0,
        )
        self.assertEqual(
            by_metric["planner_selected_observation_read_ms"][
                "selected_observation_count"
            ],
            1,
        )

    def test_scenario_summary_uses_run_level_means(self):
        per_run_rows = [
            {"scenario": "low", "metric": "read_ms", "mean": 1.0},
            {"scenario": "low", "metric": "read_ms", "mean": 2.0},
            {"scenario": "low", "metric": "read_ms", "mean": 3.0},
        ]

        result = scenario_summary(per_run_rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["number_of_runs"], 3)
        self.assertAlmostEqual(result[0]["mean_of_run_means"], 2.0)
        self.assertAlmostEqual(
            result[0]["sample_sd_of_run_means"], 1.0
        )


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestCostMeasurementAnalysis
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
