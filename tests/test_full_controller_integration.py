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

from src.control.controller import SignalController
from src.control.signal_executor import SignalExecutor


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
    print("PHASE 4 — LONG SUMO INTEGRATION TEST")
    print("=" * 110)

    belief = Belief()
    controller = SignalController()
    executor = SignalExecutor()

    # --------------------------------------------------
    # Initial simulation step
    # --------------------------------------------------

    traci.simulationStep()

    executor.initialize()

    print(
        f"Initial | "
        f"Time={traci.simulation.getTime():.0f} | "
        f"Current Action={executor.current_action}"
    )

    previous_desired_action = None
    previous_actual_action = executor.current_action

    try:

        # --------------------------------------------------
        # Run 150 steps
        # --------------------------------------------------

        for _ in range(150):

            traci.simulationStep()

            current_time = traci.simulation.getTime()

            # --------------------------------------------------
            # TRUE TRAFFIC STATE
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
            # BELIEF UPDATE
            # --------------------------------------------------

            belief.update(observation)

            # --------------------------------------------------
            # CONTROLLER
            # --------------------------------------------------

            desired_action = controller.choose_action(
                belief
            )

            # --------------------------------------------------
            # SIGNAL EXECUTOR
            # --------------------------------------------------

            actual_action = executor.apply_action(
                desired_action
            )

            # --------------------------------------------------
            # PRESSURES
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
            # LOG IMPORTANT EVENTS
            # --------------------------------------------------

            action_changed = (
                desired_action != previous_desired_action
            )

            actual_changed = (
                actual_action != previous_actual_action
            )

            # Print whenever desired or actual action changes
            if action_changed or actual_changed:

                print(
                    f"Time={current_time:.0f} | "
                    f"NS={ns_pressure} | "
                    f"EW={ew_pressure} | "
                    f"Current={executor.current_action} | "
                    f"Desired={desired_action} | "
                    f"Actual={actual_action} | "
                    f"Phase={current_phase}"
                )

            # Also print every 10 steps
            elif int(current_time) % 10 == 0:

                print(
                    f"Time={current_time:.0f} | "
                    f"NS={ns_pressure} | "
                    f"EW={ew_pressure} | "
                    f"Current={executor.current_action} | "
                    f"Desired={desired_action} | "
                    f"Phase={current_phase}"
                )

            previous_desired_action = desired_action
            previous_actual_action = actual_action

    finally:

        traci.close()

        print()
        print("SUMO connection closed")


# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":
    main()