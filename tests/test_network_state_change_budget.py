import os
import sys

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import traci

from src.acquisition.network_state_change import (
    NetworkStateChangePolicy
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
        i: NetworkStateChangePolicy(threshold=10)
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

    true_states = {}
    for _ in range(20):
        traci.simulationStep()

        current_time = traci.simulation.getTime()

        requests = {}

        for intersection_id in INTERSECTIONS:
            # State-change policy needs the
            # current traffic state.
            true_state = get_intersection_traffic_state(intersection_id)
            true_states[intersection_id] = true_state

            current_belief = Belief(
                state=true_state,
                last_observation_time=current_time
            )

            requests[intersection_id] = (
                policies[intersection_id].should_acquire(
                    current_time=current_time,
                    belief=current_belief,
                    observation_history=[]
                )
            )

        selected = manager.select_requested(requests)

        max_selected = max(
            max_selected,
            len(selected)
        )

        total_acquired += len(selected)

        for intersection_id in selected:
            state = true_states[intersection_id]

            beliefs[intersection_id].update(
                Observation(
                    traffic_state=state,
                    timestamp=current_time
                )
            )

            policies[intersection_id].record_acquisition(state)

    print("Network State-Change baseline test completed.")
    print("Maximum selected:", max_selected)
    print("Total acquired:", total_acquired)

finally:
    traci.close()