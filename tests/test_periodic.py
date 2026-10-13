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


from src.acquisition.periodic import PeriodicPolicy


def main():

    interval = 3

    policy = PeriodicPolicy(
        interval=interval
    )

    print()
    print("=" * 70)
    print("PHASE 5 — PERIODIC POLICY TEST")
    print("=" * 70)

    print(
        f"Interval: {interval} seconds"
    )

    print()

    expected_pattern = [
        True,   # 0
        False,  # 1
        False,  # 2
        True,   # 3
        False,  # 4
        False,  # 5
        True,   # 6
        False,  # 7
        False,  # 8
        True    # 9
    ]

    passed = True

    for time in range(10):

        actual = policy.should_acquire(
            current_time=time,
            belief=None,
            observation_history=[]
        )

        expected = expected_pattern[time]

        actual_decision = (
            "ACQUIRE"
            if actual
            else "SKIP"
        )

        expected_decision = (
            "ACQUIRE"
            if expected
            else "SKIP"
        )

        test_passed = (
            actual == expected
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
            f"Time: {time} | "
            f"Expected: {expected_decision:<7} | "
            f"Actual: {actual_decision}"
        )

    print()
    print("-" * 70)

    if passed:
        print(
            "PASS — Periodic policy works correctly."
        )
    else:
        print(
            "FAIL — Periodic policy has incorrect behavior."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()