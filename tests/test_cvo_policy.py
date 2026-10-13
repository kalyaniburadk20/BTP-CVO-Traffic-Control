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


from src.acquisition.acquisition_policy import (
    CVOAcquisitionPolicy
)


def main():

    print("=" * 80)
    print("PHASE 7 — CVO ACQUISITION POLICY TEST")
    print("=" * 80)

    policy = CVOAcquisitionPolicy()

    # --------------------------------------------------
    # Test 1 — Valuable observation
    # CVO > Cost
    # --------------------------------------------------

    result_1 = policy.should_acquire(
        cvo=1.0,
        total_cost=0.4
    )

    print()
    print("Test 1 — CVO > Cost")
    print("CVO: 1.0")
    print("Cost: 0.4")
    print("Expected: ACQUIRE")
    print(
        f"Got: {'ACQUIRE' if result_1 else 'SKIP'}"
    )

    assert result_1 is True

    # --------------------------------------------------
    # Test 2 — Useless observation
    # CVO < Cost
    # --------------------------------------------------

    result_2 = policy.should_acquire(
        cvo=0.0,
        total_cost=0.4
    )

    print()
    print("Test 2 — CVO < Cost")
    print("CVO: 0.0")
    print("Cost: 0.4")
    print("Expected: SKIP")
    print(
        f"Got: {'ACQUIRE' if result_2 else 'SKIP'}"
    )

    assert result_2 is False

    # --------------------------------------------------
    # Test 3 — Equal case
    # CVO == Cost
    # --------------------------------------------------

    result_3 = policy.should_acquire(
        cvo=0.4,
        total_cost=0.4
    )

    print()
    print("Test 3 — CVO == Cost")
    print("CVO: 0.4")
    print("Cost: 0.4")
    print("Expected: SKIP")
    print(
        f"Got: {'ACQUIRE' if result_3 else 'SKIP'}"
    )

    assert result_3 is False

    # --------------------------------------------------
    # Test 4 — Expensive observation
    # --------------------------------------------------

    result_4 = policy.should_acquire(
        cvo=1.0,
        total_cost=0.9
    )

    print()
    print("Test 4 — High CVO but expensive")
    print("CVO: 1.0")
    print("Cost: 0.9")
    print("Expected: ACQUIRE")
    print(
        f"Got: {'ACQUIRE' if result_4 else 'SKIP'}"
    )

    assert result_4 is True

    # --------------------------------------------------
    # Test 5 — Cost greater than CVO
    # --------------------------------------------------

    result_5 = policy.should_acquire(
        cvo=1.0,
        total_cost=1.1
    )

    print()
    print("Test 5 — Cost > CVO")
    print("CVO: 1.0")
    print("Cost: 1.1")
    print("Expected: SKIP")
    print(
        f"Got: {'ACQUIRE' if result_5 else 'SKIP'}"
    )

    assert result_5 is False

    print()
    print("=" * 80)
    print("✅ ALL CVO ACQUISITION POLICY TESTS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()