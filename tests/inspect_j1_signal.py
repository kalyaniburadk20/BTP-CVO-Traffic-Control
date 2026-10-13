import os
import sys

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import traci


SUMO_CONFIG = r".\sumo\configs\low_network.sumocfg"


traci.start([
    "sumo",
    "-c",
    SUMO_CONFIG
])

try:
    logic = traci.trafficlight.getAllProgramLogics("J5")[0]

    print("J5 phase mapping:")
    print()

    for phase_number, phase in enumerate(logic.phases):

        green_indices = [
            i
            for i, state in enumerate(phase.state)
            if state.lower() == "g"
        ]

        print(
            f"Phase {phase_number}: "
            f"duration={phase.duration}s"
        )

        print(
            f"Green link indices: {green_indices}"
        )

        print(
            f"State: {phase.state}"
        )

        print()

finally:
    traci.close()