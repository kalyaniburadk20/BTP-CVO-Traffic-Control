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


from src.acquisition.state_change import StateChangePolicy
from src.acquisition.aoi import AoIPolicy
from src.perception.traffic_state import TrafficState
from src.perception.belief import Belief


# --------------------------------------------------
# CREATE STATE
# --------------------------------------------------

def create_state(
    north,
    south,
    east,
    west
):
    return TrafficState(
        north_queue=north,
        south_queue=south,
        east_queue=east,
        west_queue=west,

        north_waiting=0.0,
        south_waiting=0.0,
        east_waiting=0.0,
        west_waiting=0.0,

        current_phase=0,
        vehicle_count=0
    )


# --------------------------------------------------
# STATE-CHANGE TEST
# --------------------------------------------------

def run_state_change_check(states):

    policy = StateChangePolicy(
        threshold=5
    )

    print()
    print("=" * 90)
    print("STATE-CHANGE POLICY — FAKE TIMELINE")
    print("=" * 90)

    for time, state in enumerate(states):

        belief = Belief(
            state=state,
            last_observation_time=time
        )

        decision = policy.should_acquire(
            current_time=time,
            belief=belief,
            observation_history=[]
        )

        decision_text = (
            "ACQUIRE"
            if decision
            else "SKIP"
        )

        reference = (
            policy._get_queue_state(
                policy.last_acquired_state
            )
        )

        print(
            f"Time={time:02d} | "
            f"State="
            f"({state.north_queue}, "
            f"{state.south_queue}, "
            f"{state.east_queue}, "
            f"{state.west_queue}) | "
            f"Decision={decision_text:<7} | "
            f"Reference={reference}"
        )


# --------------------------------------------------
# AOI TEST
# --------------------------------------------------

def run_aoi_check(states):

    policy = AoIPolicy(
        max_age=3
    )

    belief = Belief()

    print()
    print("=" * 90)
    print("AOI POLICY — FAKE TIMELINE")
    print("=" * 90)

    for time, state in enumerate(states):

        belief.state = state

        decision = policy.should_acquire(
            current_time=time,
            belief=belief,
            observation_history=[]
        )

        decision_text = (
            "ACQUIRE"
            if decision
            else "SKIP"
        )

        # Simulate acquisition so that the next
        # timestep has the correct observation age.
        if decision:
            belief.last_observation_time = time

        print(
            f"Time={time:02d} | "
            f"Decision={decision_text:<7} | "
            f"Last Observation="
            f"{belief.last_observation_time}"
        )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    # --------------------------------------------------
    # Fake traffic states
    # --------------------------------------------------

    states = [
        create_state(10, 8, 20, 5),   # S0
        create_state(10, 8, 20, 5),   # S0
        create_state(11, 8, 20, 5),   # S1
        create_state(11, 8, 20, 5),   # S1
        create_state(13, 10, 20, 5),  # S2
        create_state(13, 10, 20, 5),  # S2
        create_state(20, 15, 35, 8),  # S3
        create_state(20, 15, 35, 8),  # S3
        create_state(20, 15, 35, 8),  # S3
        create_state(21, 15, 35, 8),  # S4
        create_state(21, 15, 35, 8),  # S4
    ]

    print()
    print("#" * 90)
    print("PHASE 5 — COMMON BASELINE FAKE TIMELINE")
    print("#" * 90)

    run_state_change_check(states)

    run_aoi_check(states)

    print()
    print("#" * 90)
    print("FAKE TIMELINE TEST COMPLETE")
    print("#" * 90)


if __name__ == "__main__":
    main()