
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


SUMO_CONFIG = r".\sumo\configs\low_network.sumocfg"


traci.start([
    "sumo",
    "-c",
    SUMO_CONFIG
])

try:
    for _ in range(60):
        traci.simulationStep()

    for intersection_id in ["J1", "J2", "J3", "J4", "J5"]:

        state = get_intersection_traffic_state(
            intersection_id
        )

        print(
            f"{intersection_id}: "
            f"N={state.north_queue}, "
            f"E={state.east_queue}, "
            f"S={state.south_queue}, "
            f"W={state.west_queue}, "
            f"Phase={state.current_phase}"
        )

finally:
    traci.close()