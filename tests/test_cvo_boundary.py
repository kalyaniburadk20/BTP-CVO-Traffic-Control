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


from src.perception.traffic_state import TrafficState
from src.perception.belief import Belief
from src.control.controller import SignalController
from src.cvo.cvo import calculate_cvo


def create_belief(ns_pressure, ew_pressure):
    """
    Create a belief whose total NS and EW queue
    pressures match the requested values.
    """

    state = TrafficState(
        north_queue=ns_pressure,
        south_queue=0,
        east_queue=ew_pressure,
        west_queue=0,
        north_waiting=0.0,
        south_waiting=0.0,
        east_waiting=0.0,
        west_waiting=0.0,
        current_phase=0,
        vehicle_count=0
    )

    return Belief(
        state=state,
        last_observation_time=0.0
    )


def main():

    print("=" * 90)
    print("PHASE 6 — CVO DECISION-BOUNDARY TEST")
    print("=" * 90)

    controller = SignalController()

    margins = [1, 2, 3, 5, 10, 20, 30, 50]

    print()
    print(
        f"{'Margin':>8} | "
        f"{'Current Action':>15} | "
        f"{'Hypothetical Action':>20} | "
        f"{'CVO':>5}"
    )

    print("-" * 90)

    for margin in margins:

        # --------------------------------------------------
        # Current state:
        # NS = 20 + margin
        # EW = 20
        #
        # Therefore current action is NS_GREEN.
        # --------------------------------------------------

        current_belief = create_belief(
            ns_pressure=20 + margin,
            ew_pressure=20
        )

        # --------------------------------------------------
        # Hypothetical observation:
        # Keep NS = 20 + margin
        # Increase EW enough to cross the boundary.
        #
        # EW = 21 + margin
        # Therefore hypothetical action becomes EW_GREEN.
        # --------------------------------------------------

        hypothetical_belief = create_belief(
            ns_pressure=20 + margin,
            ew_pressure=21 + margin
        )

        current_action = controller.choose_action(
            current_belief
        )

        hypothetical_action = controller.choose_action(
            hypothetical_belief
        )

        cvo = calculate_cvo(
            current_belief,
            hypothetical_belief,
            controller
        )

        print(
            f"{margin:>8} | "
            f"{current_action:>15} | "
            f"{hypothetical_action:>20} | "
            f"{cvo:>5.1f}"
        )

        # Both actions should differ in this test.
        assert current_action == "NS_GREEN"
        assert hypothetical_action == "EW_GREEN"
        assert cvo == 1.0

    print()
    print("=" * 90)
    print(
        "All boundary-crossing cases produced CVO = 1.0"
    )
    print("=" * 90)

    print()
    print(
        "This confirms that even a small perturbation that "
        "crosses the decision boundary changes the action."
    )

    print(
        "Decision margin and CVO are related, but they are "
        "not the same quantity."
    )

    print()
    print("✅ CVO DECISION-BOUNDARY TEST PASSED")


if __name__ == "__main__":
    main()