"""Unit tests for the learned-CVO cost-sensitivity diagnostic helpers."""

import unittest

from scripts.measure_learned_cvo_cost_sensitivity import (
    percentile,
    summarize_probabilities,
)


class TestLearnedCVOCostSensitivity(unittest.TestCase):
    def test_percentile_uses_nearest_rank(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        self.assertEqual(percentile(values, 50), 3.0)
        self.assertEqual(percentile(values, 90), 5.0)
        self.assertEqual(percentile(values, 95), 5.0)

    def test_percentile_rejects_empty_values(self):
        with self.assertRaises(ValueError):
            percentile([], 95)

    def test_probability_summary_reports_distribution(self):
        summary = summarize_probabilities(
            [0.1, 0.2, 0.3, 0.4, 0.5]
        )

        self.assertEqual(summary["count"], 5.0)
        self.assertAlmostEqual(summary["mean"], 0.3)
        self.assertAlmostEqual(summary["median"], 0.3)
        self.assertAlmostEqual(summary["p90"], 0.5)
        self.assertAlmostEqual(summary["p95"], 0.5)
        self.assertAlmostEqual(summary["minimum"], 0.1)
        self.assertAlmostEqual(summary["maximum"], 0.5)

    def test_probability_summary_rejects_empty_values(self):
        with self.assertRaises(ValueError):
            summarize_probabilities([])


if __name__ == "__main__":
    unittest.main()
