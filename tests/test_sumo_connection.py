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

    try:

        while traci.simulation.getMinExpectedNumber() > 0:

            traci.simulationStep()

            current_time = traci.simulation.getTime()

            state = get_traffic_state()

            print(
                f"Time: {current_time:.0f} | "
                f"Vehicles: {state.vehicle_count} | "

                f"Queues: "
                f"N={state.north_queue} "
                f"S={state.south_queue} "
                f"E={state.east_queue} "
                f"W={state.west_queue} | "

                f"Waiting: "
                f"N={state.north_waiting:.1f} "
                f"S={state.south_waiting:.1f} "
                f"E={state.east_waiting:.1f} "
                f"W={state.west_waiting:.1f} | "

                f"Phase: {state.current_phase}"
            )

    finally:

        traci.close()

        print("SUMO connection closed")


# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":
    main()