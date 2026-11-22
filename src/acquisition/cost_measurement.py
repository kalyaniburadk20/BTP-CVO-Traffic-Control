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
        """Return count/mean/median/p95/max for each metric."""
        summary: dict[str, dict[str, float]] = {}
        metric_names = sorted({row["metric"] for row in self.records})

        for metric in metric_names:
            rows = [
                row for row in self.records
                if row["metric"] == metric
            ]

            if metric == "payload_serialization_ms":
                values = [
                    float(row["payload_bytes"]) for row in rows
                ]
                value_name = "payload_bytes"
            else:
                values = [
                    float(row["duration_ms"]) for row in rows
                ]
                value_name = "duration_ms"

            if not values:
                continue

            summary[metric] = {
                "count": float(len(values)),
                f"mean_{value_name}": float(statistics.mean(values)),
                f"median_{value_name}": float(statistics.median(values)),
                f"p95_{value_name}": self._percentile(values, 95),
                f"max_{value_name}": float(max(values)),
            }

        return summary

    def save_csv(
        self,
        output_directory: str | Path,
        scenario: str,
    ) -> Path:
        """Save a unique raw-measurement CSV; never overwrite an earlier run."""
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = output_dir / (
            f"learned_cvo_cost_{scenario}_{timestamp}.csv"
        )

        with output_path.open(
            "x", newline="", encoding="utf-8"
        ) as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=self.FIELDNAMES,
            )
            writer.writeheader()
            writer.writerows(self.records)

        return output_path
