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


from src.acquisition.always_acquire import AlwaysAcquirePolicy
from src.acquisition.periodic import PeriodicPolicy
from src.acquisition.acquisition_logger import AcquisitionLogger
from src.acquisition.metrics import (
    calculate_acquisition_rate,
    calculate_acquisition_percentage
)


# --------------------------------------------------
# TEST ONE POLICY
# --------------------------------------------------

def run_policy_check(policy, policy_name):

    output_file = os.path.join(
        PROJECT_ROOT,
        "results",
        "raw",
        f"{policy_name.lower()}_acquisition_test.csv"
    )

    logger = AcquisitionLogger(
        output_file
    )

    available_count = 10
    acquired_count = 0

    print()
    print("-" * 80)
    print(f"Testing policy: {policy_name}")
    print("-" * 80)

    for time in range(available_count):

        decision = policy.should_acquire(
            current_time=time,
            belief=None,
            observation_history=[]
        )

        if decision:
            acquired_count += 1

        logger.log(
            time=time,
            policy=policy_name,
            decision=decision
        )

        decision_text = (
            "ACQUIRE"
            if decision
            else "SKIP"
        )

        print(
            f"Time={time} | "
            f"Decision={decision_text}"
        )

    rate = calculate_acquisition_rate(
        acquired_count,
        available_count
    )

    percentage = calculate_acquisition_percentage(
        acquired_count,
        available_count
    )

    print()
    print(
        f"Available: {available_count}"
    )

    print(
        f"Acquired: {acquired_count}"
    )

    print(
        f"Acquisition Rate: {rate:.2f}"
    )

    print(
        f"Acquisition Percentage: {percentage:.1f}%"
    )

    print(
        f"Log saved to: {output_file}"
    )

    return acquired_count


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print()
    print("=" * 80)
    print("PHASE 5 — ACQUISITION METRICS + LOGGING TEST")
    print("=" * 80)

    # --------------------------------------------------
    # Always-Acquire
    # --------------------------------------------------

    always_policy = AlwaysAcquirePolicy()

    always_acquired = run_policy_check(
        always_policy,
        "Always-Acquire"
    )

    # --------------------------------------------------
    # Periodic
    # --------------------------------------------------

    periodic_policy = PeriodicPolicy(
        interval=3
    )

    periodic_acquired = run_policy_check(
        periodic_policy,
        "Periodic"
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        f"Always-Acquire → "
        f"{always_acquired}/10 acquired"
    )

    print(
        f"Periodic (3s) → "
        f"{periodic_acquired}/10 acquired"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()