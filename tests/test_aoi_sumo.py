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

from src.acquisition.aoi import AoIPolicy

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
    print("PHASE 5 — AoI + SUMO TEST")
    print("=" * 110)

    # --------------------------------------------------
    # Create modules
    # --------------------------------------------------

    policy = AoIPolicy()

    belief = Belief()

    controller = SignalController()

    executor = SignalExecutor()

    # --------------------------------------------------
    # Initialize signal executor
    # --------------------------------------------------

    traci.simulationStep()

    current_time = traci.simulation.getTime()

    executor.initialize()

    print(
        f"Initial | "
        f"Time={current_time:.0f} | "
        f"Action={executor.current_action} | "
        f"Max Age={policy.max_age}"
    )

    available_count = 0
    acquired_count = 0

    try:

        # --------------------------------------------------
        # Run 50 observation opportunities
        # --------------------------------------------------

        for _ in range(50):

            traci.simulationStep()

            current_time = traci.simulation.getTime()

            # --------------------------------------------------
            # TRUE TRAFFIC STATE
            # --------------------------------------------------

            true_state = get_traffic_state()

            available_count += 1

            # --------------------------------------------------
            # AoI POLICY
            # --------------------------------------------------

            should_acquire = policy.should_acquire(
                current_time=current_time,
                belief=belief,
                observation_history=[]
            )

            # --------------------------------------------------
            # ACQUIRE
            # --------------------------------------------------

            if should_acquire:

                acquired_count += 1

                observation = Observation(
                    traffic_state=true_state,
                    timestamp=current_time
                )

                belief.update(observation)

                desired_action = controller.choose_action(
                    belief
                )

                actual_action = executor.apply_action(
                    desired_action
                )

            # --------------------------------------------------
            # SKIP
            # --------------------------------------------------

            else:

                belief.retain()

                if belief.state is not None:

                    desired_action = (
                        controller.choose_action(
                            belief
                        )
                    )

                    actual_action = (
                        executor.apply_action(
                            desired_action
                        )
                    )

                else:

                    desired_action = None
                    actual_action = executor.current_action

            # --------------------------------------------------
            # CURRENT AoI
            # --------------------------------------------------

            age = belief.get_age(current_time)

            decision_text = (
                "ACQUIRE"
                if should_acquire
                else "SKIP"
            )

            # --------------------------------------------------
            # Print
            # --------------------------------------------------

            print(
                f"Time={current_time:.0f} | "
                f"Age={age} | "
                f"Decision={decision_text} | "
                f"Belief Time="
                f"{belief.last_observation_time} | "
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
        print("-" * 110)

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

        print(
            f"AoI max age: "
            f"{policy.max_age}"
        )

        print("=" * 110)

    finally:

        traci.close()

        print("SUMO connection closed")


if __name__ == "__main__":
    main()