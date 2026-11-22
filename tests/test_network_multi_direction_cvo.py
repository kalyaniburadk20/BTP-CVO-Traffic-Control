"""SUMO network integration test for the experimental multi-direction CVO path.

This is separate from test_network_cvo_decision.py so the original East+5
network test remains unchanged. Run from the repository root with:
    python tests/test_network_multi_direction_cvo.py
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
from src.cvo.multi_direction_cvo import MultiDirectionHypotheticalGenerator
from src.cvo.observation_cost import ObservationCost
from src.acquisition.network_planner import NetworkObservationPlanner


SUMO_CONFIG = r".\\sumo\\configs\\low_network.sumocfg"
INTERSECTIONS = ["J1", "J2", "J3", "J4", "J5"]
SIMULATION_STEPS = 10
OBSERVATION_BUDGET = 2
MAX_BELIEF_AGE = 10.0
QUEUE_DELTA = 5


def main():
    traci.start(["sumo", "-c", SUMO_CONFIG])

    try:
        beliefs = {i: Belief() for i in INTERSECTIONS}
        controllers = {
            i: NetworkSignalController(i) for i in INTERSECTIONS
        }
        executors = {
            i: NetworkSignalExecutor(i) for i in INTERSECTIONS
        }

        generator = MultiDirectionHypotheticalGenerator(
            queue_delta=QUEUE_DELTA
        )
        cost_model = ObservationCost()
        planner = NetworkObservationPlanner(
            observation_budget=OBSERVATION_BUDGET,
            max_belief_age=MAX_BELIEF_AGE,
        )

        direction_flip_counts = {
            i: {d: 0 for d in generator.DIRECTIONS}
            for i in INTERSECTIONS
        }
        selection_counts = {i: 0 for i in INTERSECTIONS}
        max_selected = 0

        # Bootstrap all beliefs with one initial observation.
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

        # Evaluate from the currently available (possibly stale) beliefs.
        for step in range(SIMULATION_STEPS):
            traci.simulationStep()
            current_time = traci.simulation.getTime()

            cvo_values = {}
            cost_values = {}
            decision_margins = {}

            for intersection_id in INTERSECTIONS:
                belief = beliefs[intersection_id]
                result = generator.evaluate(
                    belief,
                    controllers[intersection_id],
                )

                cvo_values[intersection_id] = result[
                    "multi_direction_cvo"
                ]
                decision_margins[intersection_id] = (
                    calculate_decision_margin(belief)
                )

                for direction, direction_result in result[
                    "directions"
                ].items():
                    if direction_result["cvo"] == 1.0:
                        direction_flip_counts[
                            intersection_id
                        ][direction] += 1

                # These are fixed reference/test inputs, not measured hardware costs.
                cost_result = cost_model.calculate(
                    compute_cost=40,
                    communication_cost=2,
                    latency_cost=80,
                    compute_reference=100,
                    communication_reference=5,
                    latency_reference=200,
                )
                cost_values[intersection_id] = cost_result[
                    "total_cost"
                ]

            selected = planner.select_intersections(
                cvo_values=cvo_values,
                cost_values=cost_values,
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
            for intersection_id in selected:
                selection_counts[intersection_id] += 1
                state = get_intersection_traffic_state(intersection_id)
                beliefs[intersection_id].update(
                    Observation(
                        traffic_state=state,
                        timestamp=current_time,
                    )
                )

            # Apply control using the belief that is actually available.
            for intersection_id in INTERSECTIONS:
                desired_action = controllers[
                    intersection_id
                ].choose_action(beliefs[intersection_id])
                executors[intersection_id].apply_action(desired_action)

            ages = {
                i: round(beliefs[i].get_age(current_time), 1)
                for i in INTERSECTIONS
            }
            print(
                f"t={current_time:.0f}",
                "| Selected=", selected,
                "| Multi-direction CVO=", cvo_values,
                "| Ages=", ages,
            )

        assert max_selected <= OBSERVATION_BUDGET

        print()
        print("Network multi-direction CVO integration test completed.")
        print("Maximum selected in one timestep:", max_selected)
        print("Selection counts:", selection_counts)
        print("Directional action-flip counts:", direction_flip_counts)
        print(
            "Note: +queue_delta is a controlled sensitivity probe, "
            "not a prediction of future traffic."
        )

    finally:
        traci.close()


if __name__ == "__main__":
    main()
