import os
import sys
import traci


# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.perception.traffic_state import get_traffic_state
from src.perception.observation import Observation
from src.perception.belief import Belief


# --------------------------------------------------
# SIMPLE TEST CONTROLLER
# --------------------------------------------------

def controller(belief):
    """
    Very simple test controller.

    It consumes the current belief and returns
    a signal action.

    This is NOT the final project controller.
    """

    state = belief.state

    north_south_queue = (
        state.north_queue
        + state.south_queue
    )

    east_west_queue = (
        state.east_queue
        + state.west_queue
    )

    if north_south_queue >= east_west_queue:
        return "NS_GREEN"

    return "EW_GREEN"


# --------------------------------------------------
# SUMO CONFIG
# --------------------------------------------------

CONFIG_FILE = os.path.join(
    PROJECT_ROOT,
    "sumo",
    "configs",
    "low.sumocfg"
)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    sumo_cmd = [
        "sumo",
        "-c",
        CONFIG_FILE
    ]

    traci.start(sumo_cmd)

    print("Connected to SUMO")
    print()
    print("TEST — CONTROLLER CONSUMES BELIEF")
    print("=" * 75)

    belief = Belief()

    try:

        # --------------------------------------------------
        # STEP 1 — ACQUIRE
        # --------------------------------------------------

        traci.simulationStep()

        current_time = traci.simulation.getTime()

        true_state = get_traffic_state()

        observation = Observation(
            traffic_state=true_state,
            timestamp=current_time
        )

        belief.update(observation)

        action = controller(belief)

        print(
            f"Time: {current_time:.0f} | "
            f"Decision: ACQUIRE"
        )

        print(
            f"  Belief: "
            f"N={belief.state.north_queue} "
            f"S={belief.state.south_queue} "
            f"E={belief.state.east_queue} "
            f"W={belief.state.west_queue}"
        )

        print(
            f"  Controller Action: {action}"
        )

        print("-" * 75)

        # --------------------------------------------------
        # STEP 2 — SKIP
        # --------------------------------------------------

        traci.simulationStep()

        current_time = traci.simulation.getTime()

        true_state = get_traffic_state()

        belief.retain()

        action = controller(belief)

        print(
            f"Time: {current_time:.0f} | "
            f"Decision: SKIP"
        )

        print(
            f"  Belief: "
            f"N={belief.state.north_queue} "
            f"S={belief.state.south_queue} "
            f"E={belief.state.east_queue} "
            f"W={belief.state.west_queue}"
        )

        print(
            f"  Controller Action: {action}"
        )

        print("-" * 75)

        # --------------------------------------------------
        # STEP 3 — ACQUIRE
        # --------------------------------------------------

        traci.simulationStep()

        current_time = traci.simulation.getTime()

        true_state = get_traffic_state()

        observation = Observation(
            traffic_state=true_state,
            timestamp=current_time
        )

        belief.update(observation)

        action = controller(belief)

        print(
            f"Time: {current_time:.0f} | "
            f"Decision: ACQUIRE"
        )

        print(
            f"  Belief: "
            f"N={belief.state.north_queue} "
            f"S={belief.state.south_queue} "
            f"E={belief.state.east_queue} "
            f"W={belief.state.west_queue}"
        )

        print(
            f"  Controller Action: {action}"
        )

        print("-" * 75)

    finally:

        traci.close()

        print("SUMO connection closed")


# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":
    main()