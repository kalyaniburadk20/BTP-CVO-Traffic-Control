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

from src.acquisition.state_change import (
    StateChangePolicy
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
    print("PHASE 5 — STATE-CHANGE + SUMO TEST")
    print("=" * 110)

    # --------------------------------------------------
    # Create modules
    # --------------------------------------------------

    policy = StateChangePolicy()

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
        f"Threshold={policy.threshold}"
    )

    available_count = 0
    acquired_count = 0

    try:

        # --------------------------------------------------
        # Run 100 observation opportunities
        # --------------------------------------------------

        for _ in range(100):

            traci.simulationStep()

            current_time = traci.simulation.getTime()

            # --------------------------------------------------
            # TRUE TRAFFIC STATE
            # --------------------------------------------------

            true_state = get_traffic_state()

            available_count += 1

            # --------------------------------------------------
            # STATE-CHANGE POLICY
            # --------------------------------------------------

            should_acquire = policy.should_acquire(
                current_time=current_time,
                belief=Belief(
                    state=true_state,
                    last_observation_time=current_time
                ),
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

                # Belief remains unchanged.
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
            # Current queue state
            # --------------------------------------------------

            current_queues = (
                true_state.north_queue,
                true_state.south_queue,
                true_state.east_queue,
                true_state.west_queue
            )

            # --------------------------------------------------
            # Reference queue state
            # --------------------------------------------------

            if policy.last_acquired_state is not None:

                reference_queues = (
                    policy.last_acquired_state.north_queue,
                    policy.last_acquired_state.south_queue,
                    policy.last_acquired_state.east_queue,
                    policy.last_acquired_state.west_queue
                )

                state_distance = sum(
                    abs(current - reference)
                    for current, reference
                    in zip(
                        current_queues,
                        reference_queues
                    )
                )

            else:

                state_distance = None

            decision_text = (
                "ACQUIRE"
                if should_acquire
                else "SKIP"
            )

            # --------------------------------------------------
            # Print decision
            # --------------------------------------------------

            print(
                f"Time={current_time:.0f} | "
                f"Queues={current_queues} | "
                f"Distance={state_distance} | "
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
            f"State-Change threshold: "
            f"{policy.threshold}"
        )

        print("=" * 110)

    finally:

        traci.close()

        print("SUMO connection closed")


if __name__ == "__main__":
    main()