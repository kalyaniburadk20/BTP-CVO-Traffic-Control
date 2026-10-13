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

from src.perception.belief import Belief

from src.control.network_controller import (
    NetworkSignalController
)

from src.control.network_signal_executor import (
    NetworkSignalExecutor
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

    controllers = {
        intersection_id:
            NetworkSignalController(intersection_id)
        for intersection_id in INTERSECTIONS
    }

    executors = {
        intersection_id:
            NetworkSignalExecutor(intersection_id)
        for intersection_id in INTERSECTIONS
    }

    beliefs = {
        intersection_id:
            Belief()
        for intersection_id in INTERSECTIONS
    }

    # Run 60 seconds
    for _ in range(60):

        traci.simulationStep()

        current_time = traci.simulation.getTime()

        for intersection_id in INTERSECTIONS:

            state = get_intersection_traffic_state(
                intersection_id
            )

            from src.perception.observation import (
                Observation
            )

            observation = Observation(
                traffic_state=state,
                timestamp=current_time
            )

            beliefs[intersection_id].update(
                observation
            )

            desired_action = (
                controllers[intersection_id].choose_action(
                    beliefs[intersection_id]
                )
            )

            executors[intersection_id].apply_action(
                desired_action
            )

    print("Network control test completed.")

    for intersection_id in INTERSECTIONS:

        print(
            intersection_id,
            "| Phase=",
            traci.trafficlight.getPhase(
                intersection_id
            ),
            "| Action=",
            executors[intersection_id].current_action,
            "| Belief age=",
            beliefs[intersection_id].get_age(
                traci.simulation.getTime()
            )
        )

finally:

    traci.close()