"""Non-destructive threshold sensitivity diagnostic for learned-CVO acquisition.

This diagnostic:
- loads the existing calibrated fresh-CVO artifact (no retraining);
- runs one SUMO scenario/seed using the existing 0.4 cost as the live policy;
- evaluates a candidate cost threshold on the exact same per-timestep beliefs;
- records every calibrated probability and both planners' same-input selections.

The candidate planner is counterfactual on the baseline trajectory. Its selections
are NOT applied to SUMO, so this is a one-step policy sensitivity diagnostic,
not an independent closed-loop performance comparison.
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

# Retain the existing learned-CVO integration-test cost as the baseline.
BASELINE_RAW_COSTS = {
    "compute": 40,
    "communication": 2,
    "latency": 80,
}
BASELINE_REFERENCE_COSTS = {
    "compute": 100,
    "communication": 5,
    "latency": 200,
}


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile, consistent with the cost recorder."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile of an empty list.")
    index = max(0, min(len(ordered) - 1, math.ceil(pct / 100 * len(ordered)) - 1))
    return float(ordered[index])


def summarize_probabilities(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("No learned-CVO probabilities were recorded.")
    return {
        "count": float(len(values)),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def make_planner() -> NetworkObservationPlanner:
    return NetworkObservationPlanner(
        observation_budget=OBSERVATION_BUDGET,
        max_belief_age=MAX_BELIEF_AGE,
        age_priority_on_ties=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare learned-CVO cost thresholds on the same SUMO belief trajectory."
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        choices=("low", "medium", "high"),
        default="low",
        help="Traffic scenario (default: low).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="SUMO random seed (default: 1).",
    )
    parser.add_argument(
        "--candidate-cost",
        type=float,
        default=0.68,
        help="Candidate normalized acquisition-cost threshold (default: 0.68).",
    )
    args = parser.parse_args()

    if args.seed < 0:
        parser.error("--seed must be greater than or equal to 0")
    if not 0.0 <= args.candidate_cost <= 1.0:
        parser.error("--candidate-cost must be between 0 and 1")

    project_root = PROJECT_ROOT
    sumo_config = project_root / "sumo" / "configs" / f"{args.scenario}_network.sumocfg"
    if not sumo_config.is_file():
        raise FileNotFoundError(f"SUMO network configuration not found: {sumo_config}")

    artifact = load_fresh_cvo_model()
    classification_threshold = float(artifact["threshold"])

    cost_model = ObservationCost()
    baseline_cost_result = cost_model.calculate(
        compute_cost=BASELINE_RAW_COSTS["compute"],
        communication_cost=BASELINE_RAW_COSTS["communication"],
        latency_cost=BASELINE_RAW_COSTS["latency"],
        compute_reference=BASELINE_REFERENCE_COSTS["compute"],
        communication_reference=BASELINE_REFERENCE_COSTS["communication"],
        latency_reference=BASELINE_REFERENCE_COSTS["latency"],
    )
    baseline_cost = float(baseline_cost_result["total_cost"])
    candidate_cost = float(args.candidate_cost)

    baseline_planner = make_planner()
    candidate_planner = make_planner()

    beliefs = {intersection_id: Belief() for intersection_id in INTERSECTIONS}
    controllers = {
        intersection_id: NetworkSignalController(intersection_id)
        for intersection_id in INTERSECTIONS
    }
    executors = {
        intersection_id: NetworkSignalExecutor(intersection_id)
        for intersection_id in INTERSECTIONS
    }

    probability_values: list[float] = []
    records: list[dict[str, object]] = []
    selection_counts = {
        "baseline": {intersection_id: 0 for intersection_id in INTERSECTIONS},
        "candidate": {intersection_id: 0 for intersection_id in INTERSECTIONS},
    }
    total_selected = {"baseline": 0, "candidate": 0}
    total_eligible = {"baseline": 0, "candidate": 0}

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
        print(f"Existing baseline cost: {baseline_cost:.6f}")
        print(f"Candidate cost: {candidate_cost:.6f}")
        print(f"Model classification threshold (reported only): {classification_threshold:.6f}")
        print("Running same-trajectory, one-step threshold comparison...")

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

            baseline_selected = baseline_planner.select_intersections(
                cvo_values=probabilities,
                cost_values={
                    intersection_id: baseline_cost
                    for intersection_id in INTERSECTIONS
                },
                decision_margins=margins,
                belief_ages=ages,
            )
            candidate_selected = candidate_planner.select_intersections(
                cvo_values=probabilities,
                cost_values={
                    intersection_id: candidate_cost
                    for intersection_id in INTERSECTIONS
                },
                decision_margins=margins,
                belief_ages=ages,
            )

            for intersection_id in INTERSECTIONS:
                probability = float(probabilities[intersection_id])
                age = float(ages[intersection_id])
                margin = float(margins[intersection_id])
                probability_values.append(probability)

                baseline_eligible = probability > baseline_cost
                candidate_eligible = probability > candidate_cost
                total_eligible["baseline"] += int(baseline_eligible)
                total_eligible["candidate"] += int(candidate_eligible)

                baseline_was_selected = intersection_id in baseline_selected
                candidate_was_selected = intersection_id in candidate_selected
                selection_counts["baseline"][intersection_id] += int(baseline_was_selected)
                selection_counts["candidate"][intersection_id] += int(candidate_was_selected)

                records.append({
                    "scenario": args.scenario,
                    "seed": args.seed,
                    "simulation_time": current_time,
                    "intersection_id": intersection_id,
                    "learned_cvo_probability": probability,
                    "model_classification_threshold": classification_threshold,
                    "belief_age_seconds": age,
                    "decision_margin": margin,
                    "forced_by_age": age >= MAX_BELIEF_AGE,
                    "baseline_cost_threshold": baseline_cost,
                    "candidate_cost_threshold": candidate_cost,
                    "eligible_at_baseline_cost": baseline_eligible,
                    "eligible_at_candidate_cost": candidate_eligible,
                    "selected_by_baseline_planner": baseline_was_selected,
                    "selected_by_candidate_planner": candidate_was_selected,
                })

            total_selected["baseline"] += len(baseline_selected)
            total_selected["candidate"] += len(candidate_selected)

            # Apply only the existing baseline policy to SUMO. The candidate
            # planner is counterfactual and never changes the live simulation.
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

        stats = summarize_probabilities(probability_values)
        output_dir = project_root / "results" / "processed" / "ml" / "cost_sensitivity"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = output_dir / (
            f"learned_cvo_cost_sensitivity_{args.scenario}_seed{args.seed}_{timestamp}.csv"
        )

        fieldnames = list(records[0].keys())
        with output_path.open("x", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        print("\nLearned-CVO probability distribution (5 intersections x 60 timesteps):")
        for key, value in stats.items():
            print(f"{key}: {value:.6f}")

        print("\nRaw eligibility before the network budget:")
        for name, cost in (("baseline", baseline_cost), ("candidate", candidate_cost)):
            count = total_eligible[name]
            print(
                f"{name} (cost={cost:.6f}): {count}/"
                f"{len(probability_values)} "
                f"({100.0 * count / len(probability_values):.2f}%)"
            )

        print("\nPlanner selections on the same baseline belief trajectory:")
        for name in ("baseline", "candidate"):
            print(
                f"{name}: {total_selected[name]} total selections; "
                f"per intersection={selection_counts[name]}"
            )

        print("\nSaved probability-level diagnostic:", output_path)
        print(
            "Interpretation: only baseline selections were applied to SUMO. "
            "Candidate selections are same-input counterfactuals, not an "
            "independent closed-loop run or a traffic-performance result."
        )
        print(
            "The model classification threshold is reported separately; "
            "it is not the acquisition-cost threshold."
        )
    finally:
        traci.close()


if __name__ == "__main__":
    main()
