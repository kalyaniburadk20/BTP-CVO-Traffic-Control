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


def main():

    policy = AlwaysAcquirePolicy()

    print()
    print("=" * 70)
    print("PHASE 5 — ALWAYS-ACQUIRE TEST")
    print("=" * 70)

    total_opportunities = 10
    acquired_count = 0

    for time in range(total_opportunities):

        should_acquire = policy.should_acquire(
            current_time=time,
            belief=None,
            observation_history=[]
        )

        decision = (
            "ACQUIRE"
            if should_acquire
            else "SKIP"
        )

        if should_acquire:
            acquired_count += 1

        print(
            f"Time: {time} | "
            f"Decision: {decision}"
        )

    acquisition_rate = (
        acquired_count / total_opportunities
    )

    print()
    print("-" * 70)

    print(
        f"Available observations: "
        f"{total_opportunities}"
    )

    print(
        f"Acquired observations: "
        f"{acquired_count}"
    )

    print(
        f"Acquisition Rate: "
        f"{acquisition_rate * 100:.1f}%"
    )

    print()

    if acquisition_rate == 1.0:
        print("PASS — Always-Acquire works correctly.")
    else:
        print("FAIL — Acquisition rate is not 100%.")

    print("=" * 70)


if __name__ == "__main__":
    main()