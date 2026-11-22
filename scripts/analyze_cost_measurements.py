"""Aggregate the nine seeded learned-CVO cost-measurement runs.

Run from the repository root:
    python scripts/analyze_cost_measurements.py

The raw CSVs contain five bootstrap reads at simulation time 60 and
planner-selected reads at simulation times 61..120. This script separates
those groups so bootstrap measurements do not get mixed into the per-refresh
acquisition-cost estimate.

It writes two uniquely named CSVs in the measurement directory:
- per-run metric summary
- scenario summary (mean and sample SD across seed-level run means)

No raw measurement file is modified or overwritten.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable


FILENAME_RE = re.compile(
    r"^learned_cvo_cost_(low|medium|high)_seed([1-3])_.+\.csv$"
)
REQUIRED_COLUMNS = {
    "run_seed",
    "simulation_time",
    "intersection_id",
    "metric",
    "duration_ms",
    "payload_bytes",
}
SCENARIOS = ("low", "medium", "high")
SEEDS = (1, 2, 3)
BOOTSTRAP_TIME = 60.0


def percentile_nearest_rank(values: list[float], percentile: float) -> float:
    """Compute a nearest-rank percentile without external dependencies."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile from no values.")
    index = max(
        0,
        min(
            len(ordered) - 1,
            math.ceil(percentile / 100.0 * len(ordered)) - 1,
        ),
    )
    return float(ordered[index])


def describe(values: list[float]) -> dict[str, float]:
    """Summarize one metric within one simulation run."""
    if not values:
        return {
            "sample_count": 0,
            "mean": "",
            "median": "",
            "p95": "",
            "maximum": "",
        }
    return {
        "sample_count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": percentile_nearest_rank(values, 95),
        "maximum": max(values),
    }


def load_run(path: Path) -> tuple[str, int, list[dict[str, str]]]:
    """Load and validate a single scenario/seed CSV."""
    match = FILENAME_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected seeded measurement filename: {path.name}")

    scenario = match.group(1)
    seed = int(match.group(2))

    with path.open("r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path.name} is missing CSV columns: {sorted(missing)}"
            )
        rows = list(reader)

    if not rows:
        raise ValueError(f"{path.name} contains no measurements.")

    row_seeds = {row["run_seed"] for row in rows}
    if row_seeds != {str(seed)}:
        raise ValueError(
            f"Seed metadata mismatch in {path.name}: {sorted(row_seeds)}"
        )

    return scenario, seed, rows


def build_run_metric_rows(
    path: Path,
    scenario: str,
    seed: int,
    rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    """Split bootstrap and planner-selected measurements for one run."""
    groups: dict[str, list[float]] = defaultdict(list)
    selected_by_time: dict[float, set[str]] = defaultdict(set)

    for row in rows:
        metric = row["metric"]
        simulation_time = float(row["simulation_time"])

        if metric == "model_inference_batch_ms":
            groups["model_inference_batch_ms"].append(
                float(row["duration_ms"])
            )

        elif metric == "observation_read_ms":
            if simulation_time <= BOOTSTRAP_TIME:
                groups["bootstrap_observation_read_ms"].append(
                    float(row["duration_ms"])
                )
            else:
                groups["planner_selected_observation_read_ms"].append(
                    float(row["duration_ms"])
                )
                selected_by_time[simulation_time].add(
                    row["intersection_id"]
                )

        elif metric == "payload_serialization_ms":
            if simulation_time <= BOOTSTRAP_TIME:
                groups["bootstrap_payload_serialization_ms"].append(
                    float(row["duration_ms"])
                )
                groups["bootstrap_payload_bytes"].append(
                    float(row["payload_bytes"])
                )
            else:
                groups["planner_selected_payload_serialization_ms"].append(
                    float(row["duration_ms"])
                )
                groups["planner_selected_payload_bytes"].append(
                    float(row["payload_bytes"])
                )

    output_rows: list[dict[str, object]] = []
    for metric_name in (
        "model_inference_batch_ms",
        "bootstrap_observation_read_ms",
        "planner_selected_observation_read_ms",
        "bootstrap_payload_serialization_ms",
        "planner_selected_payload_serialization_ms",
        "bootstrap_payload_bytes",
        "planner_selected_payload_bytes",
    ):
        stats = describe(groups[metric_name])
        output_rows.append({
            "scenario": scenario,
            "seed": seed,
            "source_file": path.name,
            "metric": metric_name,
            **stats,
            "selected_observation_count": sum(
                len(intersections)
                for intersections in selected_by_time.values()
            ),
            "max_selected_per_timestep": max(
                (len(intersections) for intersections in selected_by_time.values()),
                default=0,
            ),
        })

    return output_rows


def scenario_summary(
    per_run_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Calculate mean and sample SD across run-level metric means."""
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)

    for row in per_run_rows:
        value = row["mean"]
        if value != "":
            grouped[(str(row["scenario"]), str(row["metric"]))].append(
                float(value)
            )

    output: list[dict[str, object]] = []
    for (scenario, metric), run_means in sorted(grouped.items()):
        output.append({
            "scenario": scenario,
            "metric": metric,
            "number_of_runs": len(run_means),
            "mean_of_run_means": statistics.mean(run_means),
            "sample_sd_of_run_means": (
                statistics.stdev(run_means) if len(run_means) > 1 else 0.0
            ),
            "minimum_run_mean": min(run_means),
            "maximum_run_mean": max(run_means),
        })
    return output


def write_unique_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
) -> None:
    """Write a new CSV without overwriting an existing file."""
    with path.open("x", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate seeded learned-CVO cost measurement CSV files."
    )
    parser.add_argument(
        "--input-dir",
        default="results/processed/ml/cost_measurements",
        help="Directory containing the raw seeded measurement CSV files.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Measurement directory not found: {input_dir}"
        )

    files = sorted(
        path for path in input_dir.glob("learned_cvo_cost_*_seed*_*.csv")
        if FILENAME_RE.match(path.name)
    )
    if not files:
        raise FileNotFoundError(
            f"No seeded measurement CSV files found in {input_dir}"
        )

    present = {
        (FILENAME_RE.match(path.name).group(1),
         int(FILENAME_RE.match(path.name).group(2)))
        for path in files
    }
    missing = [
        f"{scenario} seed {seed}"
        for scenario in SCENARIOS
        for seed in SEEDS
        if (scenario, seed) not in present
    ]
    if missing:
        raise ValueError(
            "The nine-run set is incomplete. Missing: " + ", ".join(missing)
        )

    per_run_rows: list[dict[str, object]] = []
    for path in files:
        scenario, seed, rows = load_run(path)
        per_run_rows.extend(
            build_run_metric_rows(path, scenario, seed, rows)
        )

    scenario_rows = scenario_summary(per_run_rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    per_run_path = input_dir / (
        f"cost_measurements_per_run_summary_{timestamp}.csv"
    )
    scenario_path = input_dir / (
        f"cost_measurements_scenario_summary_{timestamp}.csv"
    )

    per_run_fields = [
        "scenario",
        "seed",
        "source_file",
        "metric",
        "sample_count",
        "mean",
        "median",
        "p95",
        "maximum",
        "selected_observation_count",
        "max_selected_per_timestep",
    ]
    scenario_fields = [
        "scenario",
        "metric",
        "number_of_runs",
        "mean_of_run_means",
        "sample_sd_of_run_means",
        "minimum_run_mean",
        "maximum_run_mean",
    ]

    write_unique_csv(per_run_path, per_run_rows, per_run_fields)
    write_unique_csv(scenario_path, scenario_rows, scenario_fields)

    print(f"Seeded raw CSV files processed: {len(files)}")
    print("Bootstrap reads are classified at simulation time <= 60.")
    print("Planner-selected reads are classified at simulation time > 60.")
    print()
    print("Scenario-level results (mean ± sample SD across run means):")
    for row in scenario_rows:
        print(
            f"{row['scenario']:6s} | {row['metric']:43s} | "
            f"{float(row['mean_of_run_means']):.6f} ± "
            f"{float(row['sample_sd_of_run_means']):.6f} "
            f"(n={row['number_of_runs']})"
        )

    print()
    print("Per-run summary:", per_run_path)
    print("Scenario summary:", scenario_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
