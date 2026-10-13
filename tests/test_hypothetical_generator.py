import os
import sys


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.perception.traffic_state import TrafficState
from src.perception.belief import Belief

from src.cvo.hypothetical import (
    HypotheticalObservationGenerator
)


def main():

    print("=" * 100)
    print("PHASE 8 — HYPOTHETICAL OBSERVATION GENERATOR TEST")
    print("=" * 100)

    # --------------------------------------------------
    # Current belief
    # --------------------------------------------------

    state = TrafficState(
        north_queue=21,
        south_queue=0,
        east_queue=20,
        west_queue=0,
        north_waiting=0.0,
        south_waiting=0.0,
        east_waiting=0.0,
        west_waiting=0.0,
        current_phase=0,
        vehicle_count=10
    )

    belief = Belief(
        state=state,
        last_observation_time=10.0
    )

    # --------------------------------------------------
    # Generate hypothetical belief
    # --------------------------------------------------

    generator = HypotheticalObservationGenerator(
        east_queue_delta=5
    )

    hypothetical_belief = generator.generate(
        belief
    )

    print()
    print(
        f"Current NS:       "
        f"{belief.state.north_queue + belief.state.south_queue}"
    )

    print(
        f"Current EW:       "
        f"{belief.state.east_queue + belief.state.west_queue}"
    )

    print(
        f"Hypothetical NS:  "
        f"{hypothetical_belief.state.north_queue + hypothetical_belief.state.south_queue}"
    )

    print(
        f"Hypothetical EW:  "
        f"{hypothetical_belief.state.east_queue + hypothetical_belief.state.west_queue}"
    )

    # --------------------------------------------------
    # Verify
    # --------------------------------------------------

    assert (
        belief.state.east_queue == 20
    )

    assert (
        hypothetical_belief.state.east_queue == 25
    )

    assert (
        hypothetical_belief.state.north_queue
        == belief.state.north_queue
    )

    assert (
        hypothetical_belief.state.south_queue
        == belief.state.south_queue
    )

    assert (
        hypothetical_belief.state.west_queue
        == belief.state.west_queue
    )

    assert (
        hypothetical_belief.last_observation_time
        == belief.last_observation_time
    )

    # Original belief must remain unchanged
    assert (
        belief.state.east_queue == 20
    )

    print()
    print(
        "Verified: original belief unchanged."
    )

    print(
        "Verified: hypothetical belief created "
        "independently."
    )

    print()
    print("=" * 100)
    print(
        "✅ HYPOTHETICAL GENERATOR TEST PASSED"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()