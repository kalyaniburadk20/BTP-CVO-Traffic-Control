import os
import sys
import csv


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
from src.cvo.decision_logger import CVODecisionLogger

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

    output_file = (
        "results/raw/cvo_cost_decision_logging_test.csv"
    )

    print("=" * 100)
    print("PHASE 7 — CVO + COST + DECISION LOGGING")
    print("=" * 100)

    controller = SignalController()
    cost_model = ObservationCost()
    policy = CVOAcquisitionPolicy()
    logger = CVODecisionLogger(output_file)

    # --------------------------------------------------
    # Common observation cost
    # --------------------------------------------------

    cost_result = cost_model.calculate(
        compute_cost=40,
        communication_cost=2,
        latency_cost=80,
        compute_reference=100,
        communication_reference=5,
        latency_reference=200
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

    decision = policy.should_acquire(
        cvo=cvo,
        total_cost=cost_result["total_cost"]
    )

    logger.log(
        time=10,
        current_action=current_action,
        hypothetical_action=hypothetical_action,
        cvo=cvo,
        cost_result=cost_result,
        decision=decision
    )

    print()
    print("-" * 100)
    print("CASE 1")
    print("-" * 100)
    print(f"CVO:      {cvo}")
    print(f"Cost:     {cost_result['total_cost']}")
    print(f"Decision: {'ACQUIRE' if decision else 'SKIP'}")

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

    decision = policy.should_acquire(
        cvo=cvo,
        total_cost=cost_result["total_cost"]
    )

    logger.log(
        time=20,
        current_action=current_action,
        hypothetical_action=hypothetical_action,
        cvo=cvo,
        cost_result=cost_result,
        decision=decision
    )

    print()
    print("-" * 100)
    print("CASE 2")
    print("-" * 100)
    print(f"CVO:      {cvo}")
    print(f"Cost:     {cost_result['total_cost']}")
    print(f"Decision: {'ACQUIRE' if decision else 'SKIP'}")

    assert cvo == 0.0
    assert cost_result["total_cost"] == 0.4
    assert decision is False

    # ==================================================
    # VERIFY CSV
    # ==================================================

    print()
    print("-" * 100)
    print("VERIFYING CSV")
    print("-" * 100)

    assert os.path.exists(output_file)

    with open(
        output_file,
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 2

    expected_columns = [
        "time",
        "current_action",
        "hypothetical_action",
        "cvo",
        "raw_compute",
        "raw_communication",
        "raw_latency",
        "normalized_compute",
        "normalized_communication",
        "normalized_latency",
        "compute_weight",
        "communication_weight",
        "latency_weight",
        "total_cost",
        "decision"
    ]

    assert reader.fieldnames == expected_columns

    # --------------------------------------------------
    # Verify row 1
    # --------------------------------------------------

    assert rows[0]["time"] == "10"
    assert rows[0]["current_action"] == "NS_GREEN"
    assert rows[0]["hypothetical_action"] == "EW_GREEN"
    assert float(rows[0]["cvo"]) == 1.0
    assert float(rows[0]["total_cost"]) == 0.4
    assert rows[0]["decision"] == "ACQUIRE"

    # --------------------------------------------------
    # Verify row 2
    # --------------------------------------------------

    assert rows[1]["time"] == "20"
    assert rows[1]["current_action"] == "NS_GREEN"
    assert rows[1]["hypothetical_action"] == "NS_GREEN"
    assert float(rows[1]["cvo"]) == 0.0
    assert float(rows[1]["total_cost"]) == 0.4
    assert rows[1]["decision"] == "SKIP"

    print()
    print(f"CSV file: {output_file}")
    print(f"Rows written: {len(rows)}")
    print(f"Columns verified: {len(reader.fieldnames)}")

    print()
    print("=" * 100)
    print("✅ CVO + COST + DECISION LOGGING TEST PASSED")
    print("=" * 100)


if __name__ == "__main__":
    main()