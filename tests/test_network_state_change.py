import os
import sys

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import traci

from src.acquisition.state_change import StateChangePolicy
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
        i: StateChangePolicy(threshold=10)
        for i in INTERSECTIONS
    }

    beliefs = {
        i: Belief()
        for i in INTERSECTIONS
    }

    acquisitions = {
        i: 0
        for i in INTERSECTIONS
    }

    # --------------------------------------------------
    # 60 TIMESTEPS
    # --------------------------------------------------

    for step in range(60):

        traci.simulationStep()

        current_time = traci.simulation.getTime()

        for intersection_id in INTERSECTIONS:

            true_state = get_intersection_traffic_state(
                intersection_id
            )

            # For this test, policy sees the true
            # current state so that state-change logic
            # can actually be evaluated.
            current_belief = Belief(
                state=true_state,
                last_observation_time=current_time
            )

            should_acquire = policies[
                intersection_id
            ].should_acquire(
                current_time=current_time,
                belief=current_belief,
                observation_history=[]
            )

            if should_acquire:

                acquisitions[
                    intersection_id
                ] += 1

                beliefs[
                    intersection_id
                ].update(
                    Observation(
                        traffic_state=true_state,
                        timestamp=current_time
                    )
                )

    print("State-change network test completed.")
    print()

    for intersection_id in INTERSECTIONS:
        print(
            intersection_id,
            "| Acquisitions:",
            acquisitions[intersection_id]
        )

finally:

    traci.close()