import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.control.controller import SignalController
from src.perception.traffic_state import TrafficState
from src.perception.belief import Belief


def create_belief(
    north_queue,
    south_queue,
    east_queue,
    west_queue,
    current_phase
):
    """
    Create a Belief object from manually supplied
    traffic-state values.
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

        current_phase=current_phase,

        vehicle_count=0
    )

    return Belief(
        state=state,
        last_observation_time=0
    )


def run_test(
    test_name,
    ns_pressure,
    ew_pressure,
    current_phase,
    expected_action
):
    """
    Run one controller test.
    """

    # Put all NS pressure into north queue.
    # Put all EW pressure into east queue.
    belief = create_belief(
        north_queue=ns_pressure,
        south_queue=0,
        east_queue=ew_pressure,
        west_queue=0,
        current_phase=current_phase
    )

    controller = SignalController()

    actual_action = controller.choose_action(belief)

    passed = actual_action == expected_action

    status = "PASS" if passed else "FAIL"

    print(
        f"{status} | "
        f"{test_name:<15} | "
        f"NS={ns_pressure:<3} | "
        f"EW={ew_pressure:<3} | "
        f"Phase={current_phase} | "
        f"Expected={expected_action:<9} | "
        f"Actual={actual_action}"
    )

    return passed


def main():

    print()
    print("=" * 100)
    print("PHASE 4 — CONTROLLER MANUAL TESTS")
    print("=" * 100)

    tests = [
        (
            "Test 1",
            30,
            10,
            0,
            "NS_GREEN"
        ),
        (
            "Test 2",
            10,
            30,
            0,
            "EW_GREEN"
        ),
        (
            "Test 3",
            20,
            20,
            0,
            "NS_GREEN"
        ),
        (
            "Test 4",
            21,
            20,
            0,
            "NS_GREEN"
        ),
        (
            "Test 5",
            20,
            21,
            0,
            "EW_GREEN"
        ),
        (
            "Test 6",
            50,
            5,
            0,
            "NS_GREEN"
        ),
        (
            "Test 7",
            5,
            50,
            0,
            "EW_GREEN"
        ),
    ]

    passed_count = 0

    for test in tests:

        passed = run_test(*test)

        if passed:
            passed_count += 1

    print("=" * 100)

    print(
        f"Passed: {passed_count}/{len(tests)}"
    )

    if passed_count == len(tests):
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")

    print("=" * 100)


if __name__ == "__main__":
    main()