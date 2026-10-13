import os
import sys
import csv
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

from src.control.controller import SignalController
from src.control.signal_executor import SignalExecutor


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

CONFIG_FILE = os.path.join(
    PROJECT_ROOT,
    "sumo",
    "configs",
    "low.sumocfg"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "raw"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "controller_decisions.csv"
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
    print("PHASE 4 — CONTROLLER DECISION LOGGING")
    print("=" * 100)

    belief = Belief()
    controller = SignalController()
    executor = SignalExecutor()

    # --------------------------------------------------
    # Initialize executor
    # --------------------------------------------------

    traci.simulationStep()

    executor.initialize()

    # --------------------------------------------------
    # CSV setup
    # --------------------------------------------------

    fieldnames = [
        "time",
        "ns_pressure",
        "ew_pressure",
        "current_phase",
        "desired_action",
        "actual_action"
    ]

    try:

        with open(
            OUTPUT_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames
            )

            writer.writeheader()

            # --------------------------------------------------
            # Run simulation
            # --------------------------------------------------

            for _ in range(150):

                traci.simulationStep()

                current_time = traci.simulation.getTime()

                # --------------------------------------------------
                # TRUE STATE
                # --------------------------------------------------

                true_state = get_traffic_state()

                # --------------------------------------------------
                # OBSERVATION
                # --------------------------------------------------

                observation = Observation(
                    traffic_state=true_state,
                    timestamp=current_time
                )

                # --------------------------------------------------
                # BELIEF
                # --------------------------------------------------

                belief.update(observation)

                # --------------------------------------------------
                # CONTROLLER
                # --------------------------------------------------

                desired_action = controller.choose_action(
                    belief
                )

                # --------------------------------------------------
                # EXECUTOR
                # --------------------------------------------------

                actual_action = executor.apply_action(
                    desired_action
                )

                # --------------------------------------------------
                # PRESSURE
                # --------------------------------------------------

                ns_pressure = (
                    belief.state.north_queue
                    + belief.state.south_queue
                )

                ew_pressure = (
                    belief.state.east_queue
                    + belief.state.west_queue
                )

                current_phase = traci.trafficlight.getPhase(
                    "J0"
                )

                # --------------------------------------------------
                # SAVE CSV ROW
                # --------------------------------------------------

                writer.writerow({
                    "time": current_time,
                    "ns_pressure": ns_pressure,
                    "ew_pressure": ew_pressure,
                    "current_phase": current_phase,
                    "desired_action": desired_action,
                    "actual_action": actual_action
                })

        print(
            f"Decision log saved to:\n{OUTPUT_FILE}"
        )

    finally:

        traci.close()

        print("SUMO connection closed")


if __name__ == "__main__":
    main()