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
# SUMO CONFIG
# --------------------------------------------------

CONFIG_FILE = os.path.join(
    PROJECT_ROOT,
    "sumo",
    "configs",
    "low.sumocfg"
)


# --------------------------------------------------
# PRINT STATE
# --------------------------------------------------

def print_state(
    time,
    decision,
    true_state,
    belief
):
    print(
        f"Time: {time:.0f} | "
        f"Decision: {decision}"
    )

    print(
        f"  True State : "
        f"N={true_state.north_queue} "
        f"S={true_state.south_queue} "
        f"E={true_state.east_queue} "
        f"W={true_state.west_queue} | "
        f"Phase={true_state.current_phase}"
    )

    print(
        f"  Belief     : "
        f"N={belief.state.north_queue} "
        f"S={belief.state.south_queue} "
        f"E={belief.state.east_queue} "
        f"W={belief.state.west_queue} | "
        f"Phase={belief.state.current_phase}"
    )

    print(
        f"  Last Obs   : "
        f"{belief.last_observation_time:.0f}"
    )

    print("-" * 75)


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
    print("TEST — TRUE TRAFFIC STATE vs BELIEF")
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

        print_state(
            current_time,
            "ACQUIRE",
            true_state,
            belief
        )

        # --------------------------------------------------
        # STEP 2 — SKIP
        # --------------------------------------------------

        traci.simulationStep()

        current_time = traci.simulation.getTime()

        true_state = get_traffic_state()

        belief.retain()

        print_state(
            current_time,
            "SKIP",
            true_state,
            belief
        )

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

        print_state(
            current_time,
            "ACQUIRE",
            true_state,
            belief
        )

        # --------------------------------------------------
        # STEP 4 — SKIP
        # --------------------------------------------------

        traci.simulationStep()

        current_time = traci.simulation.getTime()

        true_state = get_traffic_state()

        belief.retain()

        print_state(
            current_time,
            "SKIP",
            true_state,
            belief
        )

    finally:

        traci.close()

        print("SUMO connection closed")


# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":
    main()