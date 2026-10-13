import os
import sys

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import traci

from src.acquisition.periodic import (
    PeriodicPolicy
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
        i: PeriodicPolicy(interval=5)
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

    print("Network Periodic baseline test completed.")
    print("Maximum selected:", max_selected)
    print("Total acquired:", total_acquired)

finally:

    traci.close()