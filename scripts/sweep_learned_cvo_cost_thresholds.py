"""Sweep learned-CVO acquisition-cost thresholds on one baseline SUMO trajectory.

The saved calibrated model is reused without retraining. Each threshold gets an
independent planner and is evaluated on the same per-timestep beliefs. Only the
existing 0.4-cost planner's observations are applied to SUMO; all other
thresholds are counterfactual one-step comparisons, not closed-loop experiments.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, median

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import traci

from src.acquisition.network_planner import NetworkObservationPlanner
from src.control.network_controller import NetworkSignalController
from src.control.network_signal_executor import NetworkSignalExecutor
from src.cvo.cvo import calculate_decision_margin
from src.cvo.observation_cost import ObservationCost
from src.ml.inference import load_fresh_cvo_model
from src.ml.network_cvo import predict_network_fresh_cvo
from src.perception.belief import Belief
from src.perception.network_traffic_state import get_intersection_traffic_state
from src.perception.observation import Observation


INTERSECTIONS = ["J1", "J2", "J3", "J4", "J5"]
WARMUP_STEPS = 60
MEASUREMENT_STEPS = 60
OBSERVATION_BUDGET = 2
MAX_BELIEF_AGE = 10.0
BASELINE_RAW_COSTS = {"compute": 40, "communication": 2, "latency": 80}
BASELINE_REFERENCE_COSTS = {"compute": 100, "communication": 5, "latency": 200}
DEFAULT_THRESHOLDS = (
    0.05, 0.10, 0.15, 0.20, 0.22, 0.25, 0.30,
    0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.68,
)


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile."""
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty list.")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(pct / 100 * len(ordered)) - 1))
    return float(ordered[index])


def normalize_thresholds(
    requested: list[float],
    baseline_cost: float,
) -> list[float]:
    """Validate, deduplicate, and include the existing baseline threshold."""
    thresholds = [float(value) for value in requested] + [float(baseline_cost)]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("All cost thresholds must be finite values in [0, 1].")
    return sorted(set(thresholds))


def make_planner() -> NetworkObservationPlanner:
    return NetworkObservationPlanner(
        observation_budget=OBSERVATION_BUDGET,
        max_belief_age=MAX_BELIEF_AGE,
        age_priority_on_ties=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep learned-CVO cost thresholds on a shared SUMO baseline trajectory."
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        choices=("low", "medium", "high"),
        default="low",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--cost-thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
        help="Candidate normalized cost thresholds; baseline 0.4 is always included.",
    )
    args = parser.parse_args()

    if args.seed < 0:
        parser.error("--seed must be greater than or equal to 0")

    project_root = PROJECT_ROOT
    sumo_config = project_root / "sumo" / "configs" / f"{args.scenario}_network.sumocfg"
    if not sumo_config.is_file():
        raise FileNotFoundError(f"SUMO network configuration not found: {sumo_config}")

    artifact = load_fresh_cvo_model()
    model_classification_threshold = float(artifact["threshold"])

    baseline_cost_result = ObservationCost().calculate(
        compute_cost=BASELINE_RAW_COSTS["compute"],
        communication_cost=BASELINE_RAW_COSTS["communication"],
        latency_cost=BASELINE_RAW_COSTS["latency"],
        compute_reference=BASELINE_REFERENCE_COSTS["compute"],
        communication_reference=BASELINE_REFERENCE_COSTS["communication"],
        latency_reference=BASELINE_REFERENCE_COSTS["latency"],
    )
    baseline_cost = float(baseline_cost_result["total_cost"])
    thresholds = normalize_thresholds(args.cost_thresholds, baseline_cost)
    planners = {threshold: make_planner() for threshold in thresholds}

    beliefs = {intersection_id: Belief() for intersection_id in INTERSECTIONS}
    controllers = {
        intersection_id: NetworkSignalController(intersection_id)
        for intersection_id in INTERSECTIONS
    }
    executors = {
        intersection_id: NetworkSignalExecutor(intersection_id)
        for intersection_id in INTERSECTIONS
    }

    stats = {
        threshold: {
            "raw_eligible": 0,
            "selected": 0,
            "selected_forced": 0,
            "selected_by_cvo": 0,
            "max_selected_per_step": 0,
            "intersection_counts": {
                intersection_id: 0 for intersection_id in INTERSECTIONS
            },
        }
        for threshold in thresholds
    }
    probabilities_all: list[float] = []
    detail_rows: list[dict[str, object]] = []

    traci.start([
        "sumo",
        "-c",
        str(sumo_config),
        "--seed",
        str(args.seed),
    ])

    try:
        print(f"Warm-up: {WARMUP_STEPS} seconds")
        for _ in range(WARMUP_STEPS):
            traci.simulationStep()

        current_time = float(traci.simulation.getTime())
        for intersection_id in INTERSECTIONS:
            state = get_intersection_traffic_state(intersection_id)
            beliefs[intersection_id].update(
                Observation(traffic_state=state, timestamp=current_time)
            )
            executors[intersection_id].initialize()

        print(f"Scenario: {args.scenario}; seed: {args.seed}")
        print(f"Existing baseline acquisition cost: {baseline_cost:.6f}")
        print(f"Model classification threshold (not an acquisition cost): {model_classification_threshold:.6f}")
        print(f"Thresholds being compared: {thresholds}")
        print("Running same-trajectory threshold sweep...")

        for _ in range(MEASUREMENT_STEPS):
            traci.simulationStep()
            current_time = float(traci.simulation.getTime())

            probabilities = predict_network_fresh_cvo(
                beliefs=beliefs,
                current_time=current_time,
                scenario=args.scenario,
                artifact=artifact,
            )
            margins = {
                intersection_id: calculate_decision_margin(beliefs[intersection_id])
                for intersection_id in INTERSECTIONS
            }
            ages = {
                intersection_id: beliefs[intersection_id].get_age(current_time)
                for intersection_id in INTERSECTIONS
            }
            for probability in probabilities.values():
                probabilities_all.append(float(probability))

            for threshold in thresholds:
                selected = planners[threshold].select_intersections(
                    cvo_values=probabilities,
                    cost_values={
                        intersection_id: threshold
                        for intersection_id in INTERSECTIONS
                    },
                    decision_margins=margins,
                    belief_ages=ages,
                )
                threshold_stats = stats[threshold]
                raw_eligible_ids = {
                    intersection_id
                    for intersection_id, probability in probabilities.items()
                    if float(probability) > threshold
                }
                forced_selected = [
                    intersection_id for intersection_id in selected
                    if ages[intersection_id] >= MAX_BELIEF_AGE
                ]
                cvo_selected = [
                    intersection_id for intersection_id in selected
                    if ages[intersection_id] < MAX_BELIEF_AGE
                ]

                threshold_stats["raw_eligible"] += len(raw_eligible_ids)
                threshold_stats["selected"] += len(selected)
                threshold_stats["selected_forced"] += len(forced_selected)
                threshold_stats["selected_by_cvo"] += len(cvo_selected)
                threshold_stats["max_selected_per_step"] = max(
                    threshold_stats["max_selected_per_step"], len(selected)
                )
                for intersection_id in selected:
                    threshold_stats["intersection_counts"][intersection_id] += 1

                for intersection_id in INTERSECTIONS:
                    detail_rows.append({
                        "scenario": args.scenario,
                        "seed": args.seed,
                        "simulation_time": current_time,
                        "intersection_id": intersection_id,
                        "learned_cvo_probability": float(probabilities[intersection_id]),
                        "model_classification_threshold": model_classification_threshold,
                        "belief_age_seconds": float(ages[intersection_id]),
                        "decision_margin": float(margins[intersection_id]),
                        "cost_threshold": threshold,
                        "raw_cvo_eligible": intersection_id in raw_eligible_ids,
                        "forced_by_age": ages[intersection_id] >= MAX_BELIEF_AGE,
                        "selected": intersection_id in selected,
                        "selected_by_forced_age": intersection_id in forced_selected,
                        "selected_by_cvo": intersection_id in cvo_selected,
                    })

            # Apply only the existing 0.4-cost baseline policy to SUMO.
            baseline_selected = planners[baseline_cost].select_intersections(
                cvo_values=probabilities,
                cost_values={
                    intersection_id: baseline_cost
                    for intersection_id in INTERSECTIONS
                },
                decision_margins=margins,
                belief_ages=ages,
            )
            for intersection_id in baseline_selected:
                state = get_intersection_traffic_state(intersection_id)
                beliefs[intersection_id].update(
                    Observation(traffic_state=state, timestamp=current_time)
                )

            for intersection_id in INTERSECTIONS:
                action = controllers[intersection_id].choose_action(
                    beliefs[intersection_id]
                )
                executors[intersection_id].apply_action(action)

        if not probabilities_all:
            raise RuntimeError("No learned-CVO probabilities were recorded.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir = (
            project_root / "results" / "processed" / "ml" / "cost_sensitivity"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / (
            f"learned_cvo_threshold_sweep_summary_{args.scenario}_seed{args.seed}_{timestamp}.csv"
        )
        detail_path = output_dir / (
            f"learned_cvo_threshold_sweep_detail_{args.scenario}_seed{args.seed}_{timestamp}.csv"
        )

        total_probability_rows = len(probabilities_all)
        probability_mean = mean(probabilities_all)
        probability_p95 = percentile(probabilities_all, 95)
        probability_max = max(probabilities_all)

        summary_fields = [
            "scenario", "seed", "cost_threshold", "model_classification_threshold",
            "probability_count", "probability_mean", "probability_p95",
            "probability_max", "raw_eligible_count", "raw_eligibility_rate_percent",
            "total_selected", "selected_by_forced_age", "selected_by_cvo",
            "max_selected_per_step", "J1_selected", "J2_selected",
            "J3_selected", "J4_selected", "J5_selected",
        ]
        with summary_path.open("x", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=summary_fields)
            writer.writeheader()
            for threshold in thresholds:
                row = stats[threshold]
                writer.writerow({
                    "scenario": args.scenario,
                    "seed": args.seed,
                    "cost_threshold": threshold,
                    "model_classification_threshold": model_classification_threshold,
                    "probability_count": total_probability_rows,
                    "probability_mean": probability_mean,
                    "probability_p95": probability_p95,
                    "probability_max": probability_max,
                    "raw_eligible_count": row["raw_eligible"],
                    "raw_eligibility_rate_percent": (
                        100.0 * row["raw_eligible"] / total_probability_rows
                    ),
                    "total_selected": row["selected"],
                    "selected_by_forced_age": row["selected_forced"],
                    "selected_by_cvo": row["selected_by_cvo"],
                    "max_selected_per_step": row["max_selected_per_step"],
                    **{
                        f"{intersection_id}_selected":
                            row["intersection_counts"][intersection_id]
                        for intersection_id in INTERSECTIONS
                    },
                })

        detail_fields = list(detail_rows[0].keys())
        with detail_path.open("x", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=detail_fields)
            writer.writeheader()
            writer.writerows(detail_rows)

        print("\nLearned-CVO probability distribution (same baseline trajectory):")
        print(f"Count: {total_probability_rows}")
        print(f"Mean: {probability_mean:.6f}")
        print(f"P95: {probability_p95:.6f}")
        print(f"Maximum: {probability_max:.6f}")

        print("\nThreshold sweep results:")
        print(
            "cost | raw eligible | total selected | forced selected | "
            "CVO selected | J1 J2 J3 J4 J5"
        )
        for threshold in thresholds:
            row = stats[threshold]
            counts = row["intersection_counts"]
            print(
                f"{threshold:.2f} | "
                f"{row['raw_eligible']:4d}/{total_probability_rows} | "
                f"{row['selected']:4d} | "
                f"{row['selected_forced']:4d} | "
                f"{row['selected_by_cvo']:4d} | "
                + " ".join(str(counts[i]) for i in INTERSECTIONS)
            )

        print("\nSaved summary:", summary_path)
        print("Saved detailed probabilities/selections:", detail_path)
        print(
            "Only the existing baseline policy is applied to SUMO. All other "
            "threshold rows are counterfactual comparisons on the baseline "
            "trajectory; they are not separate closed-loop performance runs."
        )
        print(
            "The model classification threshold is reported separately and "
            "must not be interpreted as an acquisition-cost threshold."
        )
    finally:
        traci.close()


if __name__ == "__main__":
    main()
