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
from src.cvo.observation_cost import ObservationCost

from src.acquisition.acquisition_policy import (
    CVOAcquisitionPolicy
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

    print("=" * 100)
    print("PHASE 7 — COMPLETE CVO + COST DECISION CYCLE")
    print("=" * 100)

    controller = SignalController()

    cost_model = ObservationCost()

    policy = CVOAcquisitionPolicy()

    # --------------------------------------------------
    # Common cost setup
    # --------------------------------------------------

    cost_result = cost_model.calculate(
        compute_cost=40,
        communication_cost=2,
        latency_cost=80,
        compute_reference=100,
        communication_reference=5,
        latency_reference=200
    )

    print()
    print("Observation cost:")
    print(
        f"Normalized compute: "
        f"{cost_result['normalized_compute']}"
    )

    print(
        f"Normalized communication: "
        f"{cost_result['normalized_communication']}"
    )

    print(
        f"Normalized latency: "
        f"{cost_result['normalized_latency']}"
    )

    print(
        f"Total cost: "
        f"{cost_result['total_cost']}"
    )

    # ==================================================
    # CASE 1 — ACQUIRE
    # ==================================================

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

    # --------------------------------------------------
    # Current action
    # --------------------------------------------------

    current_action = controller.choose_action(
        current_belief
    )

    # --------------------------------------------------
    # Hypothetical action
    # --------------------------------------------------

    hypothetical_action = controller.choose_action(
        hypothetical_belief
    )

    # --------------------------------------------------
    # CVO
    # --------------------------------------------------

    cvo = calculate_cvo(
        current_belief,
        hypothetical_belief,
        controller
    )

    # --------------------------------------------------
    # Decision
    # --------------------------------------------------

    decision = policy.should_acquire(
        cvo=cvo,
        total_cost=cost_result["total_cost"]
    )

    print()
    print("-" * 100)
    print("CASE 1 — Valuable observation")
    print("-" * 100)

    print(
        f"Current action:      {current_action}"
    )

    print(
        f"Hypothetical action: {hypothetical_action}"
    )

    print(
        f"CVO:                 {cvo}"
    )

    print(
        f"Total cost:          "
        f"{cost_result['total_cost']}"
    )

    print(
        f"Decision:            "
        f"{'ACQUIRE' if decision else 'SKIP'}"
    )

    assert current_action == "NS_GREEN"
    assert hypothetical_action == "EW_GREEN"
    assert cvo == 1.0
    assert cost_result["total_cost"] == 0.4
    assert decision is True

    # ==================================================
    # CASE 2 — SKIP
    # ==================================================

    current_belief = create_belief(
        north_queue=40,
        south_queue=0,
        east_queue=10,
        west_queue=0
    )

    hypothetical_belief = create_belief(
        north_queue=42,
        south_queue=0,
        east_queue=12,
        west_queue=0
    )

    # --------------------------------------------------
    # Actions
    # --------------------------------------------------

    current_action = controller.choose_action(
        current_belief
    )

    hypothetical_action = controller.choose_action(
        hypothetical_belief
    )

    # --------------------------------------------------
    # CVO
    # --------------------------------------------------

    cvo = calculate_cvo(
        current_belief,
        hypothetical_belief,
        controller
    )

    # --------------------------------------------------
    # Decision
    # --------------------------------------------------

    decision = policy.should_acquire(
        cvo=cvo,
        total_cost=cost_result["total_cost"]
    )

    print()
    print("-" * 100)
    print("CASE 2 — Useless observation")
    print("-" * 100)

    print(
        f"Current action:      {current_action}"
    )

    print(
        f"Hypothetical action: {hypothetical_action}"
    )

    print(
        f"CVO:                 {cvo}"
    )

    print(
        f"Total cost:          "
        f"{cost_result['total_cost']}"
    )

    print(
        f"Decision:            "
        f"{'ACQUIRE' if decision else 'SKIP'}"
    )

    assert current_action == "NS_GREEN"
    assert hypothetical_action == "NS_GREEN"
    assert cvo == 0.0
    assert cost_result["total_cost"] == 0.4
    assert decision is False

    print()
    print("=" * 100)
    print("✅ COMPLETE CVO + COST DECISION CYCLE PASSED")
    print("=" * 100)


if __name__ == "__main__":
    main()