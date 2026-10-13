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


from src.cvo.decision_logger import (
    CVODecisionLogger
)


# --------------------------------------------------
# TEST OUTPUT
# --------------------------------------------------

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "raw",
    "cvo_decision_test.csv"
)


def main():

    print("=" * 90)
    print("PHASE 7 — DETAILED CVO DECISION LOGGER TEST")
    print("=" * 90)

    logger = CVODecisionLogger(
        OUTPUT_FILE
    )

    # --------------------------------------------------
    # Sample cost result
    # --------------------------------------------------

    cost_result = {
        "raw_compute": 40,
        "raw_communication": 2,
        "raw_latency": 80,

        "normalized_compute": 0.4,
        "normalized_communication": 0.4,
        "normalized_latency": 0.4,

        "compute_weight": 1 / 3,
        "communication_weight": 1 / 3,
        "latency_weight": 1 / 3,

        "total_cost": 0.4
    }

    # --------------------------------------------------
    # Case 1 — ACQUIRE
    # --------------------------------------------------

    logger.log(
        time=10,
        current_action="NS_GREEN",
        hypothetical_action="EW_GREEN",
        cvo=1.0,
        cost_result=cost_result,
        decision=True
    )

    # --------------------------------------------------
    # Case 2 — SKIP
    # --------------------------------------------------

    logger.log(
        time=20,
        current_action="NS_GREEN",
        hypothetical_action="NS_GREEN",
        cvo=0.0,
        cost_result=cost_result,
        decision=False
    )

    # --------------------------------------------------
    # Read CSV
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
    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    print(
        f"Rows written: {len(rows)}"
    )

    # --------------------------------------------------
    # Verify row count
    # --------------------------------------------------

    assert len(rows) == 2

    # --------------------------------------------------
    # Verify ACQUIRE row
    # --------------------------------------------------

    acquire_row = rows[0]

    assert acquire_row["time"] == "10"
    assert acquire_row["current_action"] == "NS_GREEN"
    assert acquire_row["hypothetical_action"] == "EW_GREEN"
    assert acquire_row["cvo"] == "1.0"

    assert acquire_row["raw_compute"] == "40"
    assert acquire_row["raw_communication"] == "2"
    assert acquire_row["raw_latency"] == "80"

    assert acquire_row["normalized_compute"] == "0.4"
    assert acquire_row["normalized_communication"] == "0.4"
    assert acquire_row["normalized_latency"] == "0.4"

    assert acquire_row["total_cost"] == "0.4"
    assert acquire_row["decision"] == "ACQUIRE"

    # --------------------------------------------------
    # Verify SKIP row
    # --------------------------------------------------

    skip_row = rows[1]

    assert skip_row["time"] == "20"
    assert skip_row["current_action"] == "NS_GREEN"
    assert skip_row["hypothetical_action"] == "NS_GREEN"
    assert skip_row["cvo"] == "0.0"

    assert skip_row["decision"] == "SKIP"

    # --------------------------------------------------
    # Print
    # --------------------------------------------------

    print()
    print("ACQUIRE row:")
    print(acquire_row)

    print()
    print("SKIP row:")
    print(skip_row)

    print()
    print("=" * 90)
    print("✅ DETAILED CVO DECISION LOGGER TEST PASSED")
    print("=" * 90)


if __name__ == "__main__":
    main()