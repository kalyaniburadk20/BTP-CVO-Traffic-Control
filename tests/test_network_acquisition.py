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

from src.control.network_signal_executor import (
    NetworkSignalExecutor
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
        i: Belief()
        for i in INTERSECTIONS
    }

    controllers = {
        i: NetworkSignalController(i)
        for i in INTERSECTIONS
    }

    executors = {
        i: NetworkSignalExecutor(i)
        for i in INTERSECTIONS
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

    acquisition_counts = {
        i: 0
        for i in INTERSECTIONS
    }

    skip_counts = {
        i: 0
        for i in INTERSECTIONS
    }

    # --------------------------------------------------
    # BOOTSTRAP: ACQUIRE ALL 5 ONCE
    # --------------------------------------------------

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

        acquisition_counts[
            intersection_id
        ] += 1

    # --------------------------------------------------
    # NETWORK ACQUISITION LOOP
    # --------------------------------------------------

    for _ in range(60):

        traci.simulationStep()

        current_time = traci.simulation.getTime()

        cvo_values = {}
        cost_values = {}
        decision_margins = {}

        # ----------------------------------------------
        # IMPORTANT:
        # Do NOT update beliefs yet.
        # CVO uses the currently available beliefs.
        # ----------------------------------------------

        for intersection_id in INTERSECTIONS:

            belief = beliefs[intersection_id]

            hypothetical_belief = (
                hypothetical_generator.generate(
                    belief
                )
            )

            controller = controllers[
                intersection_id
            ]

            cvo_values[intersection_id] = (
                calculate_cvo(
                    current_belief=belief,
                    hypothetical_belief=hypothetical_belief,
                    controller=controller
                )
            )

            decision_margins[intersection_id] = (
                calculate_decision_margin(
                    belief
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

        # ----------------------------------------------
        # CENTRALIZED NETWORK PLANNER
        # ----------------------------------------------

        print(
    "t=", current_time,
    "| CVO=", cvo_values,
    "| Ages=",
    {
        i: beliefs[i].get_age(current_time)
        for i in INTERSECTIONS
    }
)
        selected = planner.select_intersections(
            cvo_values=cvo_values,
            cost_values=cost_values,
            decision_margins=decision_margins,
            belief_ages={
        i: beliefs[i].get_age(current_time)
        for i in INTERSECTIONS
    }
        )

        selected_set = set(selected)

        # ----------------------------------------------
        # ACQUIRE OR SKIP
        # ----------------------------------------------

        for intersection_id in INTERSECTIONS:

            if intersection_id in selected_set:

                # Acquire real current observation
                state = get_intersection_traffic_state(
                    intersection_id
                )

                beliefs[intersection_id].update(
                    Observation(
                        traffic_state=state,
                        timestamp=current_time
                    )
                )

                acquisition_counts[
                    intersection_id
                ] += 1

            else:

                # Keep old belief
                beliefs[intersection_id].retain()

                skip_counts[
                    intersection_id
                ] += 1

            # Control always continues
            desired_action = (
                controllers[
                    intersection_id
                ].choose_action(
                    beliefs[intersection_id]
                )
            )

            executors[
                intersection_id
            ].apply_action(
                desired_action
            )

    print("Network acquisition test completed.")
    print()

    print(
        "Total acquisitions:",
        sum(acquisition_counts.values())
    )

    print(
        "Total skips:",
        sum(skip_counts.values())
    )

    print()

    for intersection_id in INTERSECTIONS:

        age = beliefs[
            intersection_id
        ].get_age(
            traci.simulation.getTime()
        )

        print(
            intersection_id,
            "| Acquired:",
            acquisition_counts[intersection_id],
            "| Skipped:",
            skip_counts[intersection_id],
            "| Final belief age:",
            age
        )

finally:

    traci.close()