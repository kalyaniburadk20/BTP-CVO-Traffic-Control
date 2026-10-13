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

from src.control.controller import SignalController
from src.control.signal_executor import SignalExecutor

from src.cvo.cvo import calculate_cvo
from src.cvo.observation_cost import ObservationCost
from src.cvo.decision_logger import CVODecisionLogger

from src.acquisition.acquisition_policy import (
    CVOAcquisitionPolicy
)


# --------------------------------------------------
# SUMO CONFIG
# --------------------------------------------------

CONFIG_FILE = os.path.join(
    PROJECT_ROOT,
    "sumo",
    "configs",
    "low.sumocfg"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "raw",
    "cvo_sumo_test.csv"
)


# --------------------------------------------------
# HELPER
# --------------------------------------------------

def create_hypothetical_belief(current_belief):
    """
    Create a controlled hypothetical belief.

    This is only for testing the CVO pipeline.
    It is NOT the actual future observation.
    """

    state = current_belief.state

    hypothetical_state = type(state)(
        north_queue=state.north_queue,
        south_queue=state.south_queue,
        east_queue=state.east_queue + 5,
        west_queue=state.west_queue,
        north_waiting=state.north_waiting,
        south_waiting=state.south_waiting,
        east_waiting=state.east_waiting,
        west_waiting=state.west_waiting,
        current_phase=state.current_phase,
        vehicle_count=state.vehicle_count
    )

    hypothetical_belief = Belief(
        state=hypothetical_state,
        last_observation_time=current_belief.last_observation_time
    )

    return hypothetical_belief


def main():

    sumo_cmd = [
        "sumo",
        "-c",
        CONFIG_FILE
    ]

    traci.start(sumo_cmd)

    print("Connected to SUMO")
    print()
    print("PHASE 7 — CVO + COST + SUMO TEST")
    print("=" * 100)

    # --------------------------------------------------
    # Create modules
    # --------------------------------------------------

    belief = Belief()

    controller = SignalController()

    executor = SignalExecutor()

    cost_model = ObservationCost()

    acquisition_policy = CVOAcquisitionPolicy()

    logger = CVODecisionLogger(
        OUTPUT_FILE
    )

    # --------------------------------------------------
    # Initialize signal executor
    # --------------------------------------------------

    traci.simulationStep()

    current_time = traci.simulation.getTime()

    executor.initialize()

    print(
        f"Initial | "
        f"Time={current_time:.0f} | "
        f"Action={executor.current_action}"
    )

    acquired_count = 0
    available_count = 0

    try:

        # --------------------------------------------------
        # Run 20 observation opportunities
        # --------------------------------------------------

        for _ in range(20):

            traci.simulationStep()

            current_time = traci.simulation.getTime()

            # --------------------------------------------------
            # TRUE TRAFFIC STATE
            # --------------------------------------------------

            true_state = get_traffic_state()

            available_count += 1

            # --------------------------------------------------
            # CURRENT OBSERVATION
            # --------------------------------------------------

            observation = Observation(
                traffic_state=true_state,
                timestamp=current_time
            )

            # --------------------------------------------------
            # UPDATE CURRENT BELIEF
            #
            # For this SUMO pipeline test, we acquire the
            # current observation first so that we can test
            # the complete CVO decision mechanism.
            # --------------------------------------------------

            belief.update(observation)

            # --------------------------------------------------
            # CURRENT ACTION
            # --------------------------------------------------

            current_action = controller.choose_action(
                belief
            )

            # --------------------------------------------------
            # HYPOTHETICAL BELIEF
            # --------------------------------------------------

            hypothetical_belief = (
                create_hypothetical_belief(
                    belief
                )
            )

            # --------------------------------------------------
            # HYPOTHETICAL ACTION
            # --------------------------------------------------

            hypothetical_action = (
                controller.choose_action(
                    hypothetical_belief
                )
            )

            # --------------------------------------------------
            # CVO
            # --------------------------------------------------

            cvo = calculate_cvo(
                belief,
                hypothetical_belief,
                controller
            )

            # --------------------------------------------------
            # OBSERVATION COST
            # --------------------------------------------------

            cost_result = cost_model.calculate(
                compute_cost=40,
                communication_cost=2,
                latency_cost=80,
                compute_reference=100,
                communication_reference=5,
                latency_reference=200
            )

            total_cost = cost_result["total_cost"]

            # --------------------------------------------------
            # ACQUIRE / SKIP DECISION
            # --------------------------------------------------

            should_acquire = acquisition_policy.should_acquire(
                cvo=cvo,
                total_cost=total_cost
            )

            if should_acquire:
                acquired_count += 1

                # --------------------------------------------------
                # ACQUIRE PATH
                #
                # Belief is already current in this test.
                # In the future experimental implementation,
                # the decision should be made before obtaining
                # the actual new observation.
                # --------------------------------------------------

                desired_action = controller.choose_action(
                    belief
                )

                actual_action = executor.apply_action(
                    desired_action
                )

            else:

                # --------------------------------------------------
                # SKIP PATH
                #
                # Keep existing belief and continue control.
                # --------------------------------------------------

                desired_action = controller.choose_action(
                    belief
                )

                actual_action = executor.apply_action(
                    desired_action
                )

            # --------------------------------------------------
            # LOG DECISION
            # --------------------------------------------------

            logger.log(
                time=current_time,
                current_action=current_action,
                hypothetical_action=hypothetical_action,
                cvo=cvo,
                cost_result=cost_result,
                decision=should_acquire
            )

            # --------------------------------------------------
            # PRESSURES
            # --------------------------------------------------

            ns_pressure = (
                belief.state.north_queue
                + belief.state.south_queue
            )

            ew_pressure = (
                belief.state.east_queue
                + belief.state.west_queue
            )

            print(
                f"Time={current_time:.0f} | "
                f"NS={ns_pressure} | "
                f"EW={ew_pressure} | "
                f"Current={current_action} | "
                f"Hypothetical={hypothetical_action} | "
                f"CVO={cvo:.1f} | "
                f"Cost={total_cost:.2f} | "
                f"Decision="
                f"{'ACQUIRE' if should_acquire else 'SKIP'} | "
                f"Actual={actual_action}"
            )

        # --------------------------------------------------
        # ACQUISITION RATE
        # --------------------------------------------------

        acquisition_rate = (
            acquired_count / available_count
        )

        print()
        print("-" * 100)

        print(
            f"Available observations: "
            f"{available_count}"
        )

        print(
            f"CVO acquisitions: "
            f"{acquired_count}"
        )

        print(
            f"Acquisition Rate: "
            f"{acquisition_rate * 100:.1f}%"
        )

        print(
            f"Decision log: "
            f"{OUTPUT_FILE}"
        )

        print("=" * 100)

    finally:

        traci.close()

        print("SUMO connection closed")


if __name__ == "__main__":
    main()