import os
import sys
import yaml

NETWORK_CONFIG = r".\configs\experiments\network_master.yaml"

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
    with open(
        NETWORK_CONFIG,
        "r",
        encoding="utf-8"
    ) as file:
        config = yaml.safe_load(file)

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

    hypothetical_generator = HypotheticalObservationGenerator(
        east_queue_delta=config["cvo"]["hypothetical_east_queue_delta"]
    )

    cost_model = ObservationCost()

    planner = NetworkObservationPlanner(
        observation_budget=2,
        max_belief_age=10
    )

    # --------------------------------------------------
    # BOOTSTRAP
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

    # --------------------------------------------------
    # 10-STEP NETWORK TEST
    # --------------------------------------------------

    for _ in range(10):

        traci.simulationStep()

        current_time = traci.simulation.getTime()

        cvo_values = {}
        cost_values = {}
        decision_margins = {}

        # ----------------------------------------------
        # Evaluate every intersection from its
        # currently available belief
        # ----------------------------------------------

        for intersection_id in INTERSECTIONS:

            belief = beliefs[
                intersection_id
            ]

            hypothetical_belief = (
                hypothetical_generator.generate(
                    belief
                )
            )

            controller = controllers[
                intersection_id
            ]

            cvo_values[
                intersection_id
            ] = calculate_cvo(
                current_belief=belief,
                hypothetical_belief=hypothetical_belief,
                controller=controller
            )

            decision_margins[
                intersection_id
            ] = calculate_decision_margin(
                belief
            )

            cost_result = cost_model.calculate(
    compute_cost=40,
    communication_cost=2,
    latency_cost=80,
    compute_reference=config[
        "cvo"
    ][
        "reference_costs"
    ][
        "compute"
    ],
    communication_reference=config[
        "cvo"
    ][
        "reference_costs"
    ][
        "communication"
    ],
    latency_reference=config[
        "cvo"
    ][
        "reference_costs"
    ][
        "latency"
    ]
)

            cost_values[
                intersection_id
            ] = cost_result["total_cost"]

        # ----------------------------------------------
        # CENTRALIZED PLANNER
        # ----------------------------------------------

        selected = planner.select_intersections(
            cvo_values=cvo_values,
            cost_values=cost_values,
            decision_margins=decision_margins,
            belief_ages={
                i: beliefs[i].get_age(
                    current_time
                )
                for i in INTERSECTIONS
            }
        )

        selected_set = set(selected)

        # ----------------------------------------------
        # ACQUIRE SELECTED OBSERVATIONS
        # ----------------------------------------------

        for intersection_id in INTERSECTIONS:

            if intersection_id in selected_set:

                state = get_intersection_traffic_state(
                    intersection_id
                )

                beliefs[
                    intersection_id
                ].update(
                    Observation(
                        traffic_state=state,
                        timestamp=current_time
                    )
                )

        # ----------------------------------------------
        # APPLY CONTROL USING AVAILABLE BELIEFS
        # ----------------------------------------------

        for intersection_id in INTERSECTIONS:

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

        print(
            f"t={current_time:.0f}",
            "| Selected=",
            selected,
            "| Ages=",
            {
                i: round(
                    beliefs[i].get_age(current_time),
                    1
                )
                for i in INTERSECTIONS
            }
        )

    print()
    print("Network CVO decision test completed.")

finally:

    traci.close()