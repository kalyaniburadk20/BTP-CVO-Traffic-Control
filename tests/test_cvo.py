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


def create_belief(north_queue, south_queue, east_queue, west_queue):
    """
    Create a belief containing a simple traffic state.
    """

    state = TrafficState(
        north_queue=north_queue,
        south_queue=south_queue,
        east_queue=east_queue,
        west_queue=west_queue,
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

    print("=" * 70)
    print("PHASE 6 — CVO UNIT TEST")
    print("=" * 70)

    controller = SignalController()

    # --------------------------------------------------
    # Test 1 — Same action
    # --------------------------------------------------

    current_belief = create_belief(
        north_queue=30,
        south_queue=0,
        east_queue=10,
        west_queue=0
    )

    hypothetical_belief = create_belief(
        north_queue=32,
        south_queue=0,
        east_queue=12,
        west_queue=0
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

    print("Test 1 — Same action")
    print(f"Current action:      {current_action}")
    print(f"Hypothetical action: {hypothetical_action}")
    print(f"Expected CVO: 0.0")
    print(f"Got CVO:      {cvo}")

    assert current_action == "NS_GREEN"
    assert hypothetical_action == "NS_GREEN"
    assert cvo == 0.0

    # --------------------------------------------------
    # Test 2 — Action changes
    # --------------------------------------------------

    current_belief = create_belief(
        north_queue=21,
        south_queue=0,
        east_queue=20,
        west_queue=0
    )

    hypothetical_belief = create_belief(
        north_queue=21,
        south_queue=0,
        east_queue=25,
        west_queue=0
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

    print()
    print("Test 2 — Action changes")
    print(f"Current action:      {current_action}")
    print(f"Hypothetical action: {hypothetical_action}")
    print(f"Expected CVO: 1.0")
    print(f"Got CVO:      {cvo}")

    assert current_action == "NS_GREEN"
    assert hypothetical_action == "EW_GREEN"
    assert cvo == 1.0

    # --------------------------------------------------
    # Test 3 — Extreme state
    # --------------------------------------------------

    current_belief = create_belief(
        north_queue=100,
        south_queue=0,
        east_queue=5,
        west_queue=0
    )

    hypothetical_belief = create_belief(
        north_queue=90,
        south_queue=0,
        east_queue=10,
        west_queue=0
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

    print()
    print("Test 3 — Extreme state")
    print(f"Current action:      {current_action}")
    print(f"Hypothetical action: {hypothetical_action}")
    print(f"Expected CVO: 0.0")
    print(f"Got CVO:      {cvo}")

    assert current_action == "NS_GREEN"
    assert hypothetical_action == "NS_GREEN"
    assert cvo == 0.0

    print()
    print("✅ ALL CVO UNIT TESTS PASSED")


if __name__ == "__main__":
    main()