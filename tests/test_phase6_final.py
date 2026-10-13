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

from src.cvo.cvo import (
    calculate_cvo,
    calculate_decision_margin
)


# --------------------------------------------------
# CREATE BELIEF
# --------------------------------------------------

def create_belief(
    north_queue,
    south_queue,
    east_queue,
    west_queue
):
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

    print("=" * 90)
    print("PHASE 6 — FINAL CVO VALIDATION")
    print("=" * 90)

    controller = SignalController()

    # --------------------------------------------------
    # TEST 1
    # Current: NS > EW
    # Hypothetical: still NS > EW
    # --------------------------------------------------

    current_belief = create_belief(
        30, 0, 10, 0
    )

    hypothetical_belief = create_belief(
        32, 0, 12, 0
    )

    current_action = controller.choose_action(
        current_belief
    )

    hypothetical_action = controller.choose_action(
        hypothetical_belief
    )

    margin = calculate_decision_margin(
        current_belief
    )

    cvo = calculate_cvo(
        current_belief,
        hypothetical_belief,
        controller
    )

    print()
    print("Test 1 — Same action")
    print(f"Current margin:      {margin}")
    print(f"Current action:      {current_action}")
    print(f"Hypothetical action: {hypothetical_action}")
    print(f"CVO:                 {cvo}")

    assert current_action == "NS_GREEN"
    assert hypothetical_action == "NS_GREEN"
    assert margin == 20
    assert cvo == 0.0

    # --------------------------------------------------
    # TEST 2
    # Current is close to boundary
    # Hypothetical crosses boundary
    # --------------------------------------------------

    current_belief = create_belief(
        21, 0, 20, 0
    )

    hypothetical_belief = create_belief(
        21, 0, 25, 0
    )

    current_action = controller.choose_action(
        current_belief
    )

    hypothetical_action = controller.choose_action(
        hypothetical_belief
    )

    margin = calculate_decision_margin(
        current_belief
    )

    cvo = calculate_cvo(
        current_belief,
        hypothetical_belief,
        controller
    )

    print()
    print("Test 2 — Action changes")
    print(f"Current margin:      {margin}")
    print(f"Current action:      {current_action}")
    print(f"Hypothetical action: {hypothetical_action}")
    print(f"CVO:                 {cvo}")

    assert current_action == "NS_GREEN"
    assert hypothetical_action == "EW_GREEN"
    assert margin == 1
    assert cvo == 1.0

    # --------------------------------------------------
    # TEST 3
    # Extreme state
    # --------------------------------------------------

    current_belief = create_belief(
        100, 0, 5, 0
    )

    hypothetical_belief = create_belief(
        90, 0, 10, 0
    )

    current_action = controller.choose_action(
        current_belief
    )

    hypothetical_action = controller.choose_action(
        hypothetical_belief
    )

    margin = calculate_decision_margin(
        current_belief
    )

    cvo = calculate_cvo(
        current_belief,
        hypothetical_belief,
        controller
    )

    print()
    print("Test 3 — Extreme state")
    print(f"Current margin:      {margin}")
    print(f"Current action:      {current_action}")
    print(f"Hypothetical action: {hypothetical_action}")
    print(f"CVO:                 {cvo}")

    assert current_action == "NS_GREEN"
    assert hypothetical_action == "NS_GREEN"
    assert margin == 95
    assert cvo == 0.0

    # --------------------------------------------------
    # FINAL
    # --------------------------------------------------

    print()
    print("=" * 90)
    print("✅ PHASE 6 FINAL VALIDATION PASSED")
    print("=" * 90)

    print()
    print("Validated:")
    print("Current belief → current action")
    print("Hypothetical belief → hypothetical action")
    print("Action divergence → CVO")
    print("Decision margin calculation")
    print("Same controller used in both paths")


if __name__ == "__main__":
    main()