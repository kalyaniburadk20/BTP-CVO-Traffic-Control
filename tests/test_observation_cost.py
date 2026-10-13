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


from src.cvo.observation_cost import ObservationCost


def main():

    print("=" * 80)
    print("PHASE 7 — OBSERVATION COST TEST")
    print("=" * 80)

    cost_model = ObservationCost()

    # --------------------------------------------------
    # Test 1 — Equal weights
    # --------------------------------------------------

    print()
    print("Test 1 — Default weights")

    print(
        f"Compute weight: "
        f"{cost_model.compute_weight}"
    )

    print(
        f"Communication weight: "
        f"{cost_model.communication_weight}"
    )

    print(
        f"Latency weight: "
        f"{cost_model.latency_weight}"
    )

    weight_sum = (
        cost_model.compute_weight
        + cost_model.communication_weight
        + cost_model.latency_weight
    )

    print(
        f"Weight sum: {weight_sum}"
    )

    assert abs(weight_sum - 1.0) < 1e-9

    assert abs(
        cost_model.compute_weight - 1 / 3
    ) < 1e-9

    assert abs(
        cost_model.communication_weight - 1 / 3
    ) < 1e-9

    assert abs(
        cost_model.latency_weight - 1 / 3
    ) < 1e-9

    # --------------------------------------------------
    # Test 2 — Compute normalization
    # --------------------------------------------------

    print()
    print("Test 2 — Compute normalization")

    normalized_compute = (
        cost_model.normalize_cost(
            raw_cost=40,
            reference_cost=100
        )
    )

    print(
        f"Expected: 0.4"
    )

    print(
        f"Got:      {normalized_compute}"
    )

    assert normalized_compute == 0.4

    # --------------------------------------------------
    # Test 3 — Communication normalization
    # --------------------------------------------------

    print()
    print("Test 3 — Communication normalization")

    normalized_communication = (
        cost_model.normalize_cost(
            raw_cost=2,
            reference_cost=5
        )
    )

    print(
        f"Expected: 0.4"
    )

    print(
        f"Got:      {normalized_communication}"
    )

    assert normalized_communication == 0.4

    # --------------------------------------------------
    # Test 4 — Latency normalization
    # --------------------------------------------------

    print()
    print("Test 4 — Latency normalization")

    normalized_latency = (
        cost_model.normalize_cost(
            raw_cost=80,
            reference_cost=200
        )
    )

    print(
        f"Expected: 0.4"
    )

    print(
        f"Got:      {normalized_latency}"
    )

    assert normalized_latency == 0.4

    # --------------------------------------------------
    # Test 5 — Clipping
    # --------------------------------------------------

    print()
    print("Test 5 — Normalization clipping")

    clipped = (
        cost_model.normalize_cost(
            raw_cost=150,
            reference_cost=100
        )
    )

    print(
        f"Expected: 1.0"
    )

    print(
        f"Got:      {clipped}"
    )

    assert clipped == 1.0

    # --------------------------------------------------
    # Test 6 — Total cost
    # --------------------------------------------------

    print()
    print("Test 6 — Total weighted cost")

    result = cost_model.calculate(
        compute_cost=40,
        communication_cost=2,
        latency_cost=80,
        compute_reference=100,
        communication_reference=5,
        latency_reference=200
    )

    print(
        f"Normalized compute: "
        f"{result['normalized_compute']}"
    )

    print(
        f"Normalized communication: "
        f"{result['normalized_communication']}"
    )

    print(
        f"Normalized latency: "
        f"{result['normalized_latency']}"
    )

    print(
        f"Total cost: "
        f"{result['total_cost']}"
    )

    assert result["normalized_compute"] == 0.4
    assert result["normalized_communication"] == 0.4
    assert result["normalized_latency"] == 0.4
    assert result["total_cost"] == 0.4

    # --------------------------------------------------
    # Test 7 — Raw values preserved
    # --------------------------------------------------

    print()
    print("Test 7 — Raw values preserved")

    assert result["raw_compute"] == 40
    assert result["raw_communication"] == 2
    assert result["raw_latency"] == 80

    print(
        "Raw compute:       "
        f"{result['raw_compute']}"
    )

    print(
        "Raw communication: "
        f"{result['raw_communication']}"
    )

    print(
        "Raw latency:       "
        f"{result['raw_latency']}"
    )

    print()
    print("=" * 80)
    print("✅ ALL OBSERVATION COST TESTS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()