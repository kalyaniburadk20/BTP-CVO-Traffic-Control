import os
import sys

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import traci

from src.acquisition.network_budget_manager import (
    NetworkBudgetManager
)

from src.acquisition.always_acquire import (
    AlwaysAcquirePolicy
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
        i: AlwaysAcquirePolicy()
        for i in INTERSECTIONS
    }

    beliefs = {
        i: Belief()
        for i in INTERSECTIONS
    }

    budget_manager = NetworkBudgetManager(
        observation_budget=2,
        intersection_ids=INTERSECTIONS
    )

    max_selected = 0
    total_selected = 0

    # --------------------------------------------------
    # 20 TIMESTEPS
    # --------------------------------------------------

    for _ in range(20):

        traci.simulationStep()

        current_time = traci.simulation.getTime()

        requests = {}

        for intersection_id in INTERSECTIONS:

            # Each baseline requests observation.
            requests[intersection_id] = (
                policies[intersection_id].should_acquire(
                    current_time=current_time,
                    belief=beliefs[intersection_id],
                    observation_history=[]
                )
            )

        selected = budget_manager.select_requested(
            requests
        )

        max_selected = max(
            max_selected,
            len(selected)
        )

        total_selected += len(selected)

        # Update only selected intersections.
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

    print("Network budget test completed.")
    print()
    print(
        "Maximum selected in one timestep:",
        max_selected
    )
    print(
        "Total selected across 20 timesteps:",
        total_selected
    )
    print()

    for intersection_id in INTERSECTIONS:

        print(
            intersection_id,
            "| Belief age:",
            beliefs[intersection_id].get_age(
                traci.simulation.getTime()
            )
        )

finally:

    traci.close()