import os
import sys


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.perception.traffic_state import TrafficState
from src.perception.belief import Belief
from src.control.controller import SignalController
from src.cvo.multi_direction_cvo import MultiDirectionHypotheticalGenerator


def create_belief(
    north_queue,
    south_queue,
    east_queue,
    west_queue,
    phase=0,
    observation_time=12.0,
):
    state = TrafficState(
        north_queue=north_queue,
        south_queue=south_queue,
        east_queue=east_queue,
        west_queue=west_queue,
        north_waiting=1.0,
        south_waiting=2.0,
        east_waiting=3.0,
        west_waiting=4.0,
        current_phase=phase,
        vehicle_count=25,
    )
    return Belief(
        state=state,
        last_observation_time=observation_time,
    )


def main():
    generator = MultiDirectionHypotheticalGenerator(queue_delta=5)
    controller = SignalController()

    # Test 1: Four independent hypothetical states are generated.
    belief = create_belief(10, 0, 8, 0)
    scenarios = generator.generate(belief)
    assert set(scenarios) == {"north", "south", "east", "west"}

    expected_queues = {
        "north": (15, 0, 8, 0),
        "south": (10, 5, 8, 0),
        "east": (10, 0, 13, 0),
        "west": (10, 0, 8, 5),
    }
    for direction, hypothetical in scenarios.items():
        state = hypothetical.state
        actual = (
            state.north_queue,
            state.south_queue,
            state.east_queue,
            state.west_queue,
        )
        assert actual == expected_queues[direction], (
            direction, actual, expected_queues[direction]
        )

    # Test 2: Only East/West perturbations flip this deliberately chosen state.
    # Current NS pressure=10 and EW pressure=8; +5 East or West makes EW=13.
    result = generator.evaluate(belief, controller)
    assert result["current_action"] == "NS_GREEN"
    assert result["directions"]["north"]["cvo"] == 0.0
    assert result["directions"]["south"]["cvo"] == 0.0
    assert result["directions"]["east"]["cvo"] == 1.0
    assert result["directions"]["west"]["cvo"] == 1.0
    assert result["multi_direction_cvo"] == 1.0

    # Test 3: A large pressure margin remains unchanged under all +5 probes.
    stable_belief = create_belief(50, 0, 2, 0)
    stable_result = generator.evaluate(stable_belief, controller)
    assert all(
        item["cvo"] == 0.0
        for item in stable_result["directions"].values()
    )
    assert stable_result["multi_direction_cvo"] == 0.0

    # Test 4: Hypothetical evaluation must not mutate the original belief.
    original = (
        belief.state.north_queue,
        belief.state.south_queue,
        belief.state.east_queue,
        belief.state.west_queue,
        belief.last_observation_time,
    )
    generator.evaluate(belief, controller)
    unchanged = (
        belief.state.north_queue,
        belief.state.south_queue,
        belief.state.east_queue,
        belief.state.west_queue,
        belief.last_observation_time,
    )
    assert unchanged == original

    # Test 5: Observation timestamp and non-queue fields are retained.
    for hypothetical in scenarios.values():
        assert hypothetical.last_observation_time == 12.0
        assert hypothetical.state.current_phase == 0
        assert hypothetical.state.vehicle_count == 25
        assert hypothetical.state.north_waiting == 1.0
        assert hypothetical.state.south_waiting == 2.0
        assert hypothetical.state.east_waiting == 3.0
        assert hypothetical.state.west_waiting == 4.0

    # Test 6: Invalid deltas are rejected.
    try:
        MultiDirectionHypotheticalGenerator(queue_delta=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("Negative queue_delta must be rejected.")

    try:
        MultiDirectionHypotheticalGenerator(queue_delta=1.5)
    except TypeError:
        pass
    else:
        raise AssertionError("Non-integer queue_delta must be rejected.")

    print("Multi-direction CVO tests passed: 6/6")


if __name__ == "__main__":
    main()
