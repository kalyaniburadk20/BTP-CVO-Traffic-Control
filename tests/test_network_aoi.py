import os
import sys

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import traci

from src.acquisition.aoi import (
    AoIPolicy
)

from src.acquisition.network_budget_manager import (
    NetworkBudgetManager
)

from src.perception.network_traffic_state import (
    get_intersection_traffic_state
)

from src.perception.observation import Observation
from src.perception.belief import Belief


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

    policies = {
        i: AoIPolicy(max_age=10)
        for i in INTERSECTIONS
    }

    beliefs = {
        i: Belief()
        for i in INTERSECTIONS
    }

    manager = NetworkBudgetManager(
        observation_budget=2,
        intersection_ids=INTERSECTIONS
    )

    total_acquired = 0
    max_selected = 0

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

        total_acquired += 1

    # --------------------------------------------------
    # 20 TIMESTEPS
    # --------------------------------------------------

    for _ in range(20):

        traci.simulationStep()

        current_time = traci.simulation.getTime()

        requests = {
            i: policies[i].should_acquire(
                current_time=current_time,
                belief=beliefs[i],
                observation_history=[]
            )
            for i in INTERSECTIONS
        }

        selected = manager.select_requested(
            requests
        )

        max_selected = max(
            max_selected,
            len(selected)
        )

        total_acquired += len(selected)

        for intersection_id in selected:

            state = get_intersection_traffic_state(
                intersection_id
            )

            beliefs[intersection_id].update(
                Observation(
                    traffic_state=state,
                    timestamp=current_time
                )
            )

    print("Network AoI baseline test completed.")
    print(
        "Maximum selected:",
        max_selected
    )
    print(
        "Total acquired:",
        total_acquired
    )

    print()

    for intersection_id in INTERSECTIONS:

        print(
            intersection_id,
            "| Final belief age:",
            beliefs[intersection_id].get_age(
                traci.simulation.getTime()
            )
        )

finally:

    traci.close()