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


from src.acquisition.metrics import (
    calculate_acquisition_rate,
    calculate_acquisition_percentage,
    calculate_compute_savings
)


def main():

    print("=" * 100)
    print("PHASE 7 — CVO ACQUISITION METRICS TEST")
    print("=" * 100)

    # --------------------------------------------------
    # SUMO TEST RESULT
    # --------------------------------------------------

    available_count = 20
    cvo_acquired_count = 20

    # --------------------------------------------------
    # ALWAYS-ACQUIRE REFERENCE
    # --------------------------------------------------

    always_acquired_count = 20

    # --------------------------------------------------
    # ACQUISITION RATE
    # --------------------------------------------------

    cvo_rate = calculate_acquisition_rate(
        cvo_acquired_count,
        available_count
    )

    cvo_percentage = calculate_acquisition_percentage(
        cvo_acquired_count,
        available_count
    )

    # --------------------------------------------------
    # COMPUTE SAVINGS
    # --------------------------------------------------

    compute_savings = calculate_compute_savings(
        compute_always=always_acquired_count,
        compute_policy=cvo_acquired_count
    )

    print()
    print("-" * 100)
    print("CVO RESULTS")
    print("-" * 100)

    print(
        f"Available observations: "
        f"{available_count}"
    )

    print(
        f"CVO acquisitions: "
        f"{cvo_acquired_count}"
    )

    print(
        f"Acquisition rate: "
        f"{cvo_rate:.2f}"
    )

    print(
        f"Acquisition percentage: "
        f"{cvo_percentage:.1f}%"
    )

    print(
        f"Compute savings vs Always-Acquire: "
        f"{compute_savings:.2f}"
    )

    # --------------------------------------------------
    # VERIFY
    # --------------------------------------------------

    assert cvo_rate == 1.0
    assert cvo_percentage == 100.0
    assert compute_savings == 0.0

    print()
    print("=" * 100)
    print("✅ CVO ACQUISITION METRICS TEST PASSED")
    print("=" * 100)


if __name__ == "__main__":
    main()