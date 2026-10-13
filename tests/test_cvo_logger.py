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

from src.cvo.cvo import (
    calculate_and_log_cvo
)

from src.cvo.cvo_logger import CVOLogger


# --------------------------------------------------
# OUTPUT
# --------------------------------------------------

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "raw",
    "cvo_integration_test.csv"
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

    print("=" * 80)
    print("PHASE 6 — CVO + LOGGER INTEGRATION TEST")
    print("=" * 80)

    controller = SignalController()

    logger = CVOLogger(
        OUTPUT_FILE
    )

    # --------------------------------------------------
    # CASE 1 — Same action
    # --------------------------------------------------

    current_belief = create_belief(
        north_queue=30,
        south_queue=0,
        east_queue=10,
        west_queue=0
    )

    hypothetical_belief = create_belief(
        north_queue=32,
        south_queue=0,
        east_queue=12,
        west_queue=0
    )

    cvo_1 = calculate_and_log_cvo(
        time=10,
        current_belief=current_belief,
        hypothetical_belief=hypothetical_belief,
        controller=controller,
        logger=logger
    )

    print()
    print("Case 1 — Same action")
    print(f"CVO: {cvo_1}")

    assert cvo_1 == 0.0

    # --------------------------------------------------
    # CASE 2 — Action changes
    # --------------------------------------------------

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

    cvo_2 = calculate_and_log_cvo(
        time=20,
        current_belief=current_belief,
        hypothetical_belief=hypothetical_belief,
        controller=controller,
        logger=logger
    )

    print()
    print("Case 2 — Action changes")
    print(f"CVO: {cvo_2}")

    assert cvo_2 == 1.0

    # --------------------------------------------------
    # Read CSV back
    # --------------------------------------------------

    with open(
        OUTPUT_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        rows = list(
            csv.DictReader(csv_file)
        )

    print()
    print(f"Rows written: {len(rows)}")

    # --------------------------------------------------
    # Verify number of rows
    # --------------------------------------------------

    assert len(rows) == 2

    # --------------------------------------------------
    # Verify Case 1
    # --------------------------------------------------

    assert rows[0]["time"] == "10"
    assert rows[0]["current_action"] == "NS_GREEN"
    assert rows[0]["hypothetical_action"] == "NS_GREEN"
    assert rows[0]["decision_margin"] == "20"
    assert rows[0]["cvo"] == "0.0"

    # --------------------------------------------------
    # Verify Case 2
    # --------------------------------------------------

    assert rows[1]["time"] == "20"
    assert rows[1]["current_action"] == "NS_GREEN"
    assert rows[1]["hypothetical_action"] == "EW_GREEN"
    assert rows[1]["decision_margin"] == "1"
    assert rows[1]["cvo"] == "1.0"

    # --------------------------------------------------
    # Print rows
    # --------------------------------------------------

    print()
    print("Logged Case 1:")
    print(rows[0])

    print()
    print("Logged Case 2:")
    print(rows[1])

    print()
    print("✅ CVO + LOGGER INTEGRATION TEST PASSED")


if __name__ == "__main__":
    main()