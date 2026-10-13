import os
import sys


# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.acquisition.state_change import StateChangePolicy
from src.perception.traffic_state import TrafficState
from src.perception.belief import Belief


# --------------------------------------------------
# CREATE TRAFFIC STATE
# --------------------------------------------------

def create_state(
    north,
    south,
    east,
    west
):
    """
    Create a simple TrafficState for testing.
    """

    return TrafficState(
        north_queue=north,
        south_queue=south,
        east_queue=east,
        west_queue=west,

        north_waiting=0.0,
        south_waiting=0.0,
        east_waiting=0.0,
        west_waiting=0.0,

        current_phase=0,
        vehicle_count=0
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    threshold = 5

    policy = StateChangePolicy(
        threshold=threshold
    )

    print()
    print("=" * 85)
    print("PHASE 5 — STATE-CHANGE POLICY TEST")
    print("=" * 85)

    print(
        f"Threshold: {threshold}"
    )

    print()

    # --------------------------------------------------
    # Initial state
    # --------------------------------------------------

    states = [
        create_state(10, 8, 20, 5),   # initial
        create_state(10, 9, 20, 5),   # distance = 1
        create_state(11, 10, 21, 6),  # distance = 4
        create_state(13, 12, 23, 7),  # distance = 10
        create_state(13, 12, 24, 7),  # distance = 1
        create_state(20, 15, 35, 8),  # large change
    ]

    expected = [
        True,
        False,
        False,
        True,
        False,
        True
    ]

    previous_reference = None

    passed = True

    for index, (
        state,
        expected_decision
    ) in enumerate(
        zip(states, expected)
    ):

        belief = Belief(
            state=state,
            last_observation_time=index
        )

        actual_decision = policy.should_acquire(
            current_time=index,
            belief=belief,
            observation_history=[]
        )

        decision_text = (
            "ACQUIRE"
            if actual_decision
            else "SKIP"
        )

        expected_text = (
            "ACQUIRE"
            if expected_decision
            else "SKIP"
        )

        test_passed = (
            actual_decision == expected_decision
        )

        if not test_passed:
            passed = False

        status = (
            "PASS"
            if test_passed
            else "FAIL"
        )

        print(
            f"{status} | "
            f"Step={index} | "
            f"Expected={expected_text:<7} | "
            f"Actual={decision_text:<7} | "
            f"Reference={policy._get_queue_state(policy.last_acquired_state)}"
        )

    print()
    print("-" * 85)

    if passed:
        print(
            "PASS — State-Change policy works correctly."
        )
    else:
        print(
            "FAIL — State-Change policy has incorrect behavior."
        )

    print("=" * 85)


if __name__ == "__main__":
    main()