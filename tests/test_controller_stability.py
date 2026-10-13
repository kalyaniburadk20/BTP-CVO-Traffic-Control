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


from src.control.controller import SignalController
from src.perception.traffic_state import TrafficState
from src.perception.belief import Belief


# --------------------------------------------------
# CREATE BELIEF
# --------------------------------------------------

def create_belief(
    ns_pressure,
    ew_pressure,
    current_phase
):
    """
    Create a manually constructed belief.
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

        current_phase=current_phase,
        vehicle_count=0
    )

    return Belief(
        state=state,
        last_observation_time=0
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    controller = SignalController()

    belief = create_belief(
        ns_pressure=20,
        ew_pressure=20,
        current_phase=0
    )

    print()
    print("=" * 80)
    print("PHASE 4 — CONTROLLER STABILITY TEST")
    print("=" * 80)

    actions = []

    # --------------------------------------------------
    # Repeatedly evaluate the same balanced state
    # --------------------------------------------------

    for i in range(10):

        action = controller.choose_action(
            belief
        )

        actions.append(action)

        print(
            f"Test {i + 1:02d} | "
            f"NS=20 | "
            f"EW=20 | "
            f"Current Phase=0 | "
            f"Action={action}"
        )

    # --------------------------------------------------
    # Check stability
    # --------------------------------------------------

    stable = all(
        action == "NS_GREEN"
        for action in actions
    )

    print()
    print("-" * 80)

    if stable:
        print("PASS — Controller remains stable.")
    else:
        print("FAIL — Controller is oscillating.")

    print("=" * 80)


# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":
    main()