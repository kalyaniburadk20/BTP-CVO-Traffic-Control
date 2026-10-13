import os
import sys
import traci

# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.perception.traffic_state import get_traffic_state
from src.perception.observation import Observation
from src.perception.belief import Belief

from src.acquisition.always_acquire import AlwaysAcquirePolicy
from src.acquisition.periodic import PeriodicPolicy
from src.acquisition.state_change import StateChangePolicy
from src.acquisition.aoi import AoIPolicy

from src.acquisition.acquisition_logger import AcquisitionLogger


# --------------------------------------------------
# FILES
# --------------------------------------------------

SUMO_CONFIG = os.path.join(
    PROJECT_ROOT,
    "sumo",
    "configs",
    "low.sumocfg"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "raw"
)


# --------------------------------------------------
# RUN ONE POLICY
# --------------------------------------------------

def run_policy(policy_name, policy, output_file):

    print()
    print("=" * 80)
    print(f"Running policy: {policy_name}")
    print("=" * 80)

    sumo_cmd = [
        "sumo",
        "-c",
        SUMO_CONFIG
    ]

    traci.start(sumo_cmd)

    logger = AcquisitionLogger(output_file)

    belief = Belief()

    available_count = 0
    acquired_count = 0

    try:

        # --------------------------------------------------
        # Run 50 observation opportunities
        # --------------------------------------------------

        for _ in range(50):

            traci.simulationStep()

            current_time = traci.simulation.getTime()

            true_state = get_traffic_state()

            available_count += 1

            # --------------------------------------------------
            # Decide acquisition
            # --------------------------------------------------

            if isinstance(policy, StateChangePolicy):

                # State-Change needs the current state
                # to compare against its last acquired state.
                policy_belief = Belief(
                    state=true_state,
                    last_observation_time=current_time
                )

                decision = policy.should_acquire(
                    current_time=current_time,
                    belief=policy_belief,
                    observation_history=[]
                )

            else:

                decision = policy.should_acquire(
                    current_time=current_time,
                    belief=belief,
                    observation_history=[]
                )

            # --------------------------------------------------
            # ACQUIRE
            # --------------------------------------------------

            if decision:

                acquired_count += 1

                observation = Observation(
                    traffic_state=true_state,
                    timestamp=current_time
                )

                belief.update(observation)

            # --------------------------------------------------
            # SKIP
            # --------------------------------------------------

            else:

                belief.retain()

            # --------------------------------------------------
            # Save decision
            # --------------------------------------------------

            logger.log(
                time=current_time,
                policy=policy_name,
                decision=decision
            )

        # --------------------------------------------------
        # Summary
        # --------------------------------------------------

        acquisition_rate = (
            acquired_count / available_count
            if available_count > 0
            else 0.0
        )

        print(
            f"Available: {available_count}"
        )

        print(
            f"Acquired: {acquired_count}"
        )

        print(
            f"Acquisition Rate: "
            f"{acquisition_rate * 100:.1f}%"
        )

        print(
            f"Saved to: {output_file}"
        )

    finally:

        traci.close()

        print("SUMO connection closed")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    # --------------------------------------------------
    # Policy definitions
    # --------------------------------------------------

    policies = [
        (
            "Always-Acquire",
            AlwaysAcquirePolicy(),
            "always_acquire_raw.csv"
        ),
        (
            "Periodic",
            PeriodicPolicy(),
            "periodic_raw.csv"
        ),
        (
            "State-Change",
            StateChangePolicy(),
            "state_change_raw.csv"
        ),
        (
            "AoI",
            AoIPolicy(),
            "aoi_raw.csv"
        )
    ]

    # --------------------------------------------------
    # Run each policy separately
    # --------------------------------------------------

    for policy_name, policy, filename in policies:

        output_file = os.path.join(
            RESULTS_DIR,
            filename
        )

        run_policy(
            policy_name=policy_name,
            policy=policy,
            output_file=output_file
        )

    print()
    print("=" * 80)
    print("ALL BASELINE ACQUISITION LOGS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()