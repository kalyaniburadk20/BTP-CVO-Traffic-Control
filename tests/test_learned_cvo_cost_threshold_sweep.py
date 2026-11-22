"""Unit tests for learned-CVO cost threshold sweep helpers."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.sweep_learned_cvo_cost_thresholds import (
    normalize_thresholds,
    percentile,
)


class TestLearnedCVOCostThresholdSweep(unittest.TestCase):
    def test_thresholds_are_sorted_deduplicated_and_include_baseline(self):
        thresholds = normalize_thresholds(
            [0.30, 0.10, 0.30, 0.68],
            baseline_cost=0.40,
        )

        self.assertEqual(thresholds, [0.10, 0.30, 0.40, 0.68])

    def test_threshold_validation_rejects_values_outside_unit_interval(self):
        for invalid in (-0.01, 1.01, float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    normalize_thresholds([invalid], baseline_cost=0.40)

    def test_percentile_uses_nearest_rank(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        self.assertEqual(percentile(values, 50), 3.0)
        self.assertEqual(percentile(values, 95), 5.0)

    def test_percentile_rejects_empty_values(self):
        with self.assertRaises(ValueError):
            percentile([], 95)


if __name__ == "__main__":
    unittest.main()
