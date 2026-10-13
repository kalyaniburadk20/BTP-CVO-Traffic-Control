import os
import sys


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.acquisition.metrics import (
    calculate_communication_data,
    calculate_compute_calls,
    calculate_compute_savings,
    create_latency_record
)


def main():

    print("=" * 70)
    print("PHASE 5 — COST METRICS TEST")
    print("=" * 70)

    # --------------------------------------------------
    # Communication
    # --------------------------------------------------

    communication = calculate_communication_data(
        acquired_count=40,
        observation_size_bytes=100 * 1024
    )

    print(
        f"Communication data: "
        f"{communication} bytes"
    )

    assert communication == 40 * 100 * 1024

    # --------------------------------------------------
    # Compute calls
    # --------------------------------------------------

    compute = calculate_compute_calls(40)

    print(
        f"Compute calls: {compute}"
    )

    assert compute == 40

    # --------------------------------------------------
    # Compute savings
    # --------------------------------------------------

    savings = calculate_compute_savings(
        compute_always=100,
        compute_policy=40
    )

    print(
        f"Compute savings: "
        f"{savings * 100:.1f}%"
    )

    assert savings == 0.60

    # --------------------------------------------------
    # Latency record
    # --------------------------------------------------

    latency = create_latency_record(
        acquisition_time=0.001,
        processing_time=0.002,
        controller_decision_time=0.0005
    )

    print(
        f"Latency record: {latency}"
    )

    assert latency["acquisition_time"] == 0.001
    assert latency["processing_time"] == 0.002
    assert (
        latency["controller_decision_time"]
        == 0.0005
    )

    print()
    print("✅ ALL COST METRIC TESTS PASSED")


if __name__ == "__main__":
    main()