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

from src.control.network_controller import (
    NetworkSignalController
)

from src.cvo.cvo import (
    calculate_cvo,
    calculate_decision_margin
)

from src.cvo.hypothetical import (
    HypotheticalObservationGenerator
)

from src.cvo.observation_cost import (
    ObservationCost
)

from src.acquisition.network_planner import (
    NetworkObservationPlanner
)


SUMO_CONFIG = r".\sumo\configs\low_network.sumocfg"

INTERSECTIONS = [
    "J1",
    "J2",
    "J3",
    "J4",
    "J5"
]


traci.start([
    "sumo",
    "-c",
    SUMO_CONFIG
])

try:

    beliefs = {
        intersection_id: Belief()
        for intersection_id in INTERSECTIONS
    }

    controllers = {
        intersection_id:
            NetworkSignalController(intersection_id)
        for intersection_id in INTERSECTIONS
    }

    hypothetical_generator = (
        HypotheticalObservationGenerator(
            east_queue_delta=5
        )
    )

    cost_model = ObservationCost()

    planner = NetworkObservationPlanner(
        observation_budget=2
    )

    # Bootstrap
    traci.simulationStep()

    current_time = traci.simulation.getTime()

    for intersection_id in INTERSECTIONS:

        state = get_intersection_traffic_state(
            intersection_id
        )

        beliefs[intersection_id].update(
            Observation(
                traffic_state=state,
                timestamp=current_time
            )
        )

    selection_counts = {
        intersection_id: 0
        for intersection_id in INTERSECTIONS
    }

    max_selected = 0
    total_selected = 0

    # --------------------------------------------------
    # 60 TIMESTEPS
    # --------------------------------------------------

    for _ in range(60):

        traci.simulationStep()

        current_time = traci.simulation.getTime()

        cvo_values = {}
        cost_values = {}
        decision_margins = {}

        # Read and update all current beliefs
        for intersection_id in INTERSECTIONS:

            state = get_intersection_traffic_state(
                intersection_id
            )

            beliefs[intersection_id].update(
                Observation(
                    traffic_state=state,
                    timestamp=current_time
                )
            )

            hypothetical_belief = (
                hypothetical_generator.generate(
                    beliefs[intersection_id]
                )
            )

            cvo_values[intersection_id] = calculate_cvo(
                current_belief=beliefs[intersection_id],
                hypothetical_belief=hypothetical_belief,
                controller=controllers[intersection_id]
            )

            decision_margins[intersection_id] = (
                calculate_decision_margin(
                    beliefs[intersection_id]
                )
            )

            cost_result = cost_model.calculate(
                compute_cost=40,
                communication_cost=2,
                latency_cost=80,
                compute_reference=100,
                communication_reference=5,
                latency_reference=200
            )

            cost_values[intersection_id] = (
                cost_result["total_cost"]
            )

        selected = planner.select_intersections(
            cvo_values=cvo_values,
            cost_values=cost_values,
            decision_margins=decision_margins
        )

        selected_count = len(selected)

        total_selected += selected_count

        max_selected = max(
            max_selected,
            selected_count
        )

        for intersection_id in selected:
            selection_counts[
                intersection_id
            ] += 1

    print("Dynamic network planner test completed.")
    print()

    print(
        "Maximum intersections selected in one timestep:",
        max_selected
    )

    print(
        "Total selections across 60 timesteps:",
        total_selected
    )

    print()

    for intersection_id in INTERSECTIONS:

        print(
            intersection_id,
            "| Selected:",
            selection_counts[intersection_id],
            "times"
        )

finally:

    traci.close()