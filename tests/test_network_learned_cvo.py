"""SUMO integration test for learned Fresh-Queue CVO acquisition.

Run from repository root:
    python tests/test_network_learned_cvo.py

This is an experimental comparison path. It uses the saved calibrated model,
compares its predicted action-divergence probability against normalized
observation cost, and applies the existing network observation budget.
It does not retrain or overwrite any ML artifact.
"""

import os
import sys

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import traci

from src.perception.network_traffic_state import (
    get_intersection_traffic_state
)
from src.perception.observation import Observation
from src.perception.belief import Belief
from src.control.network_controller import NetworkSignalController
from src.control.network_signal_executor import NetworkSignalExecutor
from src.cvo.cvo import calculate_decision_margin
from src.cvo.observation_cost import ObservationCost
from src.acquisition.network_planner import NetworkObservationPlanner
from src.ml.inference import load_fresh_cvo_model
from src.ml.network_cvo import predict_network_fresh_cvo


SUMO_CONFIG = r".\sumo\configs\low_network.sumocfg"
SCENARIO = "low"
INTERSECTIONS = ["J1", "J2", "J3", "J4", "J5"]
SIMULATION_STEPS = 20  # Long enough to observe multiple forced-refresh batches.

OBSERVATION_BUDGET = 2
MAX_BELIEF_AGE = 10.0

# These are the same reference/test inputs used by the earlier integration
# checks. They are not measured hardware costs.
RAW_COSTS = {
    "compute": 40,
    "communication": 2,
    "latency": 80,
}
REFERENCE_COSTS = {
    "compute": 100,
    "communication": 5,
    "latency": 200,
}


def main():
    # Load the trained/calibrated artifact once; reuse it at each timestep.
    learned_cvo_artifact = load_fresh_cvo_model()

    traci.start(["sumo", "-c", SUMO_CONFIG])

    try:
        beliefs = {i: Belief() for i in INTERSECTIONS}
        controllers = {
            i: NetworkSignalController(i) for i in INTERSECTIONS
        }
        executors = {
            i: NetworkSignalExecutor(i) for i in INTERSECTIONS
        }

        cost_model = ObservationCost()
        cost_result = cost_model.calculate(
            compute_cost=RAW_COSTS["compute"],
            communication_cost=RAW_COSTS["communication"],
            latency_cost=RAW_COSTS["latency"],
            compute_reference=REFERENCE_COSTS["compute"],
            communication_reference=REFERENCE_COSTS["communication"],
            latency_reference=REFERENCE_COSTS["latency"],
        )
        normalized_observation_cost = cost_result["total_cost"]

        planner = NetworkObservationPlanner(
            observation_budget=OBSERVATION_BUDGET,
            max_belief_age=MAX_BELIEF_AGE,
            age_priority_on_ties=True,
        )

        selection_counts = {i: 0 for i in INTERSECTIONS}
        max_selected = 0
        total_selected = 0

        # Bootstrap each belief with one actual SUMO observation.
        traci.simulationStep()
        current_time = traci.simulation.getTime()

        for intersection_id in INTERSECTIONS:
            state = get_intersection_traffic_state(intersection_id)
            beliefs[intersection_id].update(
                Observation(
                    traffic_state=state,
                    timestamp=current_time,
                )
            )

        print("Loaded calibrated fresh-CVO model.")
        print("Calibration:", learned_cvo_artifact["calibration_method"])
        print(
            "Model classification threshold (reported, not used as the "
            "acquisition-cost threshold):",
            round(float(learned_cvo_artifact["threshold"]), 3),
        )
        print(
            "Normalized observation cost used for acquisition:",
            round(normalized_observation_cost, 4),
        )
        print()

        for _ in range(SIMULATION_STEPS):
            traci.simulationStep()
            current_time = traci.simulation.getTime()

            # Predict from the currently available beliefs. Unselected
            # intersections intentionally retain older beliefs.
            learned_cvo_probabilities = predict_network_fresh_cvo(
                beliefs=beliefs,
                current_time=current_time,
                scenario=SCENARIO,
                artifact=learned_cvo_artifact,
            )

            decision_margins = {
                i: calculate_decision_margin(beliefs[i])
                for i in INTERSECTIONS
            }

            selected = planner.select_intersections(
                cvo_values=learned_cvo_probabilities,
                cost_values={
                    i: normalized_observation_cost
                    for i in INTERSECTIONS
                },
                decision_margins=decision_margins,
                belief_ages={
                    i: beliefs[i].get_age(current_time)
                    for i in INTERSECTIONS
                },
            )

            if len(selected) > OBSERVATION_BUDGET:
                raise AssertionError(
                    f"Observation budget exceeded at t={current_time}: "
                    f"{selected}"
                )

            max_selected = max(max_selected, len(selected))
            total_selected += len(selected)

            # Acquire only selected intersections.
            for intersection_id in selected:
                selection_counts[intersection_id] += 1
                state = get_intersection_traffic_state(intersection_id)
                beliefs[intersection_id].update(
                    Observation(
                        traffic_state=state,
                        timestamp=current_time,
                    )
                )

            # Control continues using each intersection's available belief.
            for intersection_id in INTERSECTIONS:
                desired_action = controllers[
                    intersection_id
                ].choose_action(beliefs[intersection_id])
                executors[intersection_id].apply_action(desired_action)

            print(
                f"t={current_time:.0f}",
                "| Learned CVO probabilities=",
                {
                    i: round(learned_cvo_probabilities[i], 3)
                    for i in INTERSECTIONS
                },
                "| Selected=",
                selected,
                "| Ages=",
                {
                    i: round(beliefs[i].get_age(current_time), 1)
                    for i in INTERSECTIONS
                },
            )

        assert max_selected <= OBSERVATION_BUDGET

        print()
        print("Network learned-CVO integration test completed.")
        print("Maximum selected in one timestep:", max_selected)
        print("Total observations selected:", total_selected)
        print("Selection counts:", selection_counts)
        print(
            "Note: probabilities estimate action divergence under the "
            "training label definition; they are not traffic forecasts "
            "or guarantees of traffic improvement."
        )
        print(
            "Note: observation costs are reference/test values, not "
            "measured compute, communication, or hardware latency."
        )

    finally:
        traci.close()


if __name__ == "__main__":
    main()
