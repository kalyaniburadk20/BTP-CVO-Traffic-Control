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

from src.acquisition.always_acquire import (
    AlwaysAcquirePolicy
)

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


def main():

    sumo_cmd = [
        "sumo",
        "-c",
        CONFIG_FILE
    ]

    traci.start(sumo_cmd)

    print("Connected to SUMO")
    print()
    print("PHASE 5 — ALWAYS-ACQUIRE + SUMO TEST")
    print("=" * 100)

    # --------------------------------------------------
    # Create modules
    # --------------------------------------------------

    policy = AlwaysAcquirePolicy()

    belief = Belief()

    controller = SignalController()

    executor = SignalExecutor()

    # --------------------------------------------------
    # Initialize SUMO signal executor
    # --------------------------------------------------

    traci.simulationStep()

    current_time = traci.simulation.getTime()

    executor.initialize()

    print(
        f"Initial | "
        f"Time={current_time:.0f} | "
        f"Action={executor.current_action}"
    )

    acquired_count = 0
    available_count = 0

    try:

        # --------------------------------------------------
        # Run 20 observation opportunities
        # --------------------------------------------------

        for _ in range(20):

            traci.simulationStep()

            current_time = traci.simulation.getTime()

            # --------------------------------------------------
            # TRUE TRAFFIC STATE
            # --------------------------------------------------

            true_state = get_traffic_state()

            # Every simulation step = observation available
            available_count += 1

            # --------------------------------------------------
            # ACQUISITION POLICY
            # --------------------------------------------------

            should_acquire = policy.should_acquire(
                current_time=current_time,
                belief=belief,
                observation_history=[]
            )

            # --------------------------------------------------
            # ALWAYS-ACQUIRE SHOULD ALWAYS BE TRUE
            # --------------------------------------------------

            if should_acquire:

                acquired_count += 1

                # Create observation
                observation = Observation(
                    traffic_state=true_state,
                    timestamp=current_time
                )

                # Update belief
                belief.update(observation)

                # Controller
                desired_action = controller.choose_action(
                    belief
                )

                # Signal execution
                actual_action = executor.apply_action(
                    desired_action
                )

                # Calculate pressures
                ns_pressure = (
                    belief.state.north_queue
                    + belief.state.south_queue
                )

                ew_pressure = (
                    belief.state.east_queue
                    + belief.state.west_queue
                )

            else:

                # This branch should never happen
                # for Always-Acquire.
                desired_action = None
                actual_action = executor.current_action
                ns_pressure = None
                ew_pressure = None

            print(
                f"Time={current_time:.0f} | "
                f"Available=YES | "
                f"Decision="
                f"{'ACQUIRE' if should_acquire else 'SKIP'} | "
                f"NS={ns_pressure} | "
                f"EW={ew_pressure} | "
                f"Desired={desired_action} | "
                f"Actual={actual_action}"
            )

        # --------------------------------------------------
        # Acquisition rate
        # --------------------------------------------------

        acquisition_rate = (
            acquired_count / available_count
        )

        print()
        print("-" * 100)

        print(
            f"Available observations: "
            f"{available_count}"
        )

        print(
            f"Acquired observations: "
            f"{acquired_count}"
        )

        print(
            f"Acquisition Rate: "
            f"{acquisition_rate * 100:.1f}%"
        )

        if acquisition_rate == 1.0:
            print(
                "PASS — Always-Acquire maintained 100% acquisition."
            )
        else:
            print(
                "FAIL — Acquisition rate is below 100%."
            )

        print("=" * 100)

    finally:

        traci.close()

        print("SUMO connection closed")


if __name__ == "__main__":
    main()