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
    print("TEST A — ALWAYS ACQUIRE")
    print("=" * 70)

    # Initially no belief
    belief = Belief()

    try:

        # Run only first 10 simulation steps
        for _ in range(10):

            traci.simulationStep()

            current_time = traci.simulation.getTime()

            # ------------------------------------------
            # TRUE TRAFFIC STATE
            # ------------------------------------------

            true_state = get_traffic_state()

            # ------------------------------------------
            # OBSERVATION ACQUIRED
            # ------------------------------------------

            observation = Observation(
                traffic_state=true_state,
                timestamp=current_time
            )

            # ------------------------------------------
            # UPDATE BELIEF
            # ------------------------------------------

            belief.update(observation)

            # ------------------------------------------
            # PRINT
            # ------------------------------------------

            print(
                f"Time: {current_time:.0f} | "
                f"Observation: ACQUIRED | "
                f"Belief Time: "
                f"{belief.last_observation_time:.0f} | "
                f"Belief Queues: "
                f"N={belief.state.north_queue} "
                f"S={belief.state.south_queue} "
                f"E={belief.state.east_queue} "
                f"W={belief.state.west_queue}"
            )

    finally:

        traci.close()

        print()
        print("SUMO connection closed")


# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":
    main()