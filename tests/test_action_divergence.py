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


from src.cvo.action_divergence import action_divergence


def main():

    print("=" * 70)
    print("PHASE 6 — ACTION DIVERGENCE TEST")
    print("=" * 70)

    # --------------------------------------------------
    # Test 1: Same action
    # --------------------------------------------------

    result_1 = action_divergence(
        "NS_GREEN",
        "NS_GREEN"
    )

    print(
        "Test 1: NS_GREEN → NS_GREEN"
    )
    print(
        f"Expected: 0.0 | Got: {result_1}"
    )

    assert result_1 == 0.0

    # --------------------------------------------------
    # Test 2: Different action
    # --------------------------------------------------

    result_2 = action_divergence(
        "NS_GREEN",
        "EW_GREEN"
    )

    print(
        "Test 2: NS_GREEN → EW_GREEN"
    )
    print(
        f"Expected: 1.0 | Got: {result_2}"
    )

    assert result_2 == 1.0

    # --------------------------------------------------
    # Test 3: Reverse direction
    # --------------------------------------------------

    result_3 = action_divergence(
        "EW_GREEN",
        "NS_GREEN"
    )

    print(
        "Test 3: EW_GREEN → NS_GREEN"
    )
    print(
        f"Expected: 1.0 | Got: {result_3}"
    )

    assert result_3 == 1.0

    # --------------------------------------------------
    # Test 4: Same EW action
    # --------------------------------------------------

    result_4 = action_divergence(
        "EW_GREEN",
        "EW_GREEN"
    )

    print(
        "Test 4: EW_GREEN → EW_GREEN"
    )
    print(
        f"Expected: 0.0 | Got: {result_4}"
    )

    assert result_4 == 0.0

    print()
    print("✅ ALL ACTION DIVERGENCE TESTS PASSED")


if __name__ == "__main__":
    main()