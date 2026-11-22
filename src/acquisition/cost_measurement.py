"""Measure local simulation, inference, and serialized-observation costs.

Important scope:
- model_inference_batch_ms measures local batch inference wall-clock time.
- observation_read_ms measures the Python/TraCI call used to read SUMO state.
- payload_bytes measures the compact JSON size of the state representation.
- payload_serialization_ms measures local JSON serialization time.

SUMO runs locally here. Payload bytes are a communication-volume proxy; this
module does not claim to measure real network-transfer time or physical sensor
latency.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


class CostMeasurementRecorder:
    """Collect raw runtime measurements without changing acquisition policy."""

    FIELDNAMES = [
        "run_seed",
        "simulation_time",
        "intersection_id",
        "metric",
        "duration_ms",
        "payload_bytes",
    ]

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    @staticmethod
    def _elapsed_ms(start_ns: int) -> float:
        return (time.perf_counter_ns() - start_ns) / 1_000_000.0

    def measure_model_inference(
        self,
        simulation_time: float,
        operation: Callable[[], Any],
    ) -> Any:
        """Time one complete network-batch model inference call."""
        start_ns = time.perf_counter_ns()
        result = operation()
        elapsed_ms = self._elapsed_ms(start_ns)

        self.records.append({
            "simulation_time": float(simulation_time),
            "intersection_id": "NETWORK",
            "metric": "model_inference_batch_ms",
            "duration_ms": elapsed_ms,
            "payload_bytes": "",
        })
        return result

    def measure_observation_read(
        self,
        simulation_time: float,
        intersection_id: str,
        operation: Callable[[], Any],
    ) -> Any:
        """Time one observation read and separately measure its JSON payload.

        The operation should return a traffic-state object exposing to_dict().
        Observation reads are only timed when the caller actually performs one,
        e.g. at bootstrap or for an intersection selected by the planner.
        """
        start_ns = time.perf_counter_ns()
        state = operation()
        read_elapsed_ms = self._elapsed_ms(start_ns)

        self.records.append({
            "simulation_time": float(simulation_time),
            "intersection_id": str(intersection_id),
            "metric": "observation_read_ms",
            "duration_ms": read_elapsed_ms,
            "payload_bytes": "",
        })

        if not hasattr(state, "to_dict"):
            raise TypeError("Observation state must provide to_dict().")

        serialization_start_ns = time.perf_counter_ns()
        payload = json.dumps(
            state.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        serialization_elapsed_ms = self._elapsed_ms(
            serialization_start_ns
        )

        self.records.append({
            "simulation_time": float(simulation_time),
            "intersection_id": str(intersection_id),
            "metric": "payload_serialization_ms",
            "duration_ms": serialization_elapsed_ms,
            "payload_bytes": len(payload),
        })
        return state

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        ordered = sorted(values)
        if not ordered:
            return float("nan")
        index = max(
            0,
            min(
                len(ordered) - 1,
                math.ceil((percentile / 100.0) * len(ordered)) - 1,
            ),
        )
        return float(ordered[index])

    def summarize(self) -> dict[str, dict[str, float]]:
        """Return runtime and payload-size statistics for each metric."""
        summary: dict[str, dict[str, float]] = {}
        metric_names = sorted({row["metric"] for row in self.records})

        for metric in metric_names:
            rows = [
                row for row in self.records
                if row["metric"] == metric
            ]
            durations = [
                float(row["duration_ms"]) for row in rows
            ]
            stats = {
                "count": float(len(durations)),
                "mean_duration_ms": float(statistics.mean(durations)),
                "median_duration_ms": float(statistics.median(durations)),
                "p95_duration_ms": self._percentile(durations, 95),
                "max_duration_ms": float(max(durations)),
            }

            payload_sizes = [
                float(row["payload_bytes"])
                for row in rows
                if row["payload_bytes"] != ""
            ]
            if payload_sizes:
                stats.update({
                    "mean_payload_bytes": float(
                        statistics.mean(payload_sizes)
                    ),
                    "median_payload_bytes": float(
                        statistics.median(payload_sizes)
                    ),
                    "p95_payload_bytes": self._percentile(
                        payload_sizes, 95
                    ),
                    "max_payload_bytes": float(max(payload_sizes)),
                })

            summary[metric] = stats

        return summary

    def save_csv(
        self,
        output_directory: str | Path,
        scenario: str,
        seed: int | None = None,
    ) -> Path:
        """Save a unique raw-measurement CSV; never overwrite an earlier run.

        The seed is included both in the filename and every CSV row so raw
        measurements remain traceable to their SUMO simulation run.
        """
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        seed_label = f"seed{seed}" if seed is not None else "seed_unspecified"
        output_path = output_dir / (
            f"learned_cvo_cost_{scenario}_{seed_label}_{timestamp}.csv"
        )

        with output_path.open(
            "x", newline="", encoding="utf-8"
        ) as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=self.FIELDNAMES,
            )
            writer.writeheader()
            for record in self.records:
                writer.writerow({
                    "run_seed": seed if seed is not None else "",
                    **record,
                })

        return output_path
