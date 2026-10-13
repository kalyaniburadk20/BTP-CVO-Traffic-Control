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


from src.acquisition.aoi import AoIPolicy
from src.perception.belief import Belief


def main():

    max_age = 3

    policy = AoIPolicy(
        max_age=max_age
    )

    print()
    print("=" * 75)
    print("PHASE 5 — AOI / FRESHNESS POLICY TEST")
    print("=" * 75)

    print(
        f"Maximum age: {max_age} seconds"
    )

    print()

    # --------------------------------------------------
    # Expected behavior
    #
    # t=0 -> ACQUIRE  (first observation)
    # t=1 -> SKIP
    # t=2 -> SKIP
    # t=3 -> ACQUIRE
    # t=4 -> SKIP
    # t=5 -> SKIP
    # t=6 -> ACQUIRE
    # --------------------------------------------------

    expected = {
        0: True,
        1: False,
        2: False,
        3: True,
        4: False,
        5: False,
        6: True
    }

    # No observation has been acquired initially
    belief = Belief()

    passed = True

    for time in range(7):

        actual = policy.should_acquire(
            current_time=time,
            belief=belief,
            observation_history=[]
        )

        expected_decision = expected[time]

        actual_text = (
            "ACQUIRE"
            if actual
            else "SKIP"
        )

        expected_text = (
            "ACQUIRE"
            if expected_decision
            else "SKIP"
        )

        test_passed = (
            actual == expected_decision
        )

        if not test_passed:
            passed = False

        status = (
            "PASS"
            if test_passed
            else "FAIL"
        )

        # --------------------------------------------------
        # Simulate an acquisition
        # --------------------------------------------------

        if actual:

            belief.last_observation_time = time

        print(
            f"{status} | "
            f"Time={time} | "
            f"Expected={expected_text:<7} | "
            f"Actual={actual_text:<7} | "
            f"Last Observation="
            f"{belief.last_observation_time}"
        )

    print()
    print("-" * 75)

    if passed:
        print(
            "PASS — AoI policy works correctly."
        )
    else:
        print(
            "FAIL — AoI policy has incorrect behavior."
        )

    print("=" * 75)


if __name__ == "__main__":
    main()