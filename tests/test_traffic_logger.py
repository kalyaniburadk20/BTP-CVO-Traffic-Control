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


from src.experiments.traffic_logger import TrafficLogger
from src.perception.traffic_state import TrafficState


def main():

    output_file = os.path.join(
        PROJECT_ROOT,
        "results",
        "raw",
        "traffic_logger_test.csv"
    )

    print("=" * 100)
    print("PHASE 8 — TRAFFIC LOGGER TEST")
    print("=" * 100)

    logger = TrafficLogger(
        output_file
    )

    state = TrafficState(
        north_queue=2,
        south_queue=1,
        east_queue=3,
        west_queue=0,
        north_waiting=4.0,
        south_waiting=2.0,
        east_waiting=6.0,
        west_waiting=0.0,
        current_phase=0,
        vehicle_count=7
    )

    logger.log(
        time=60,
        state=state,
        arrived_this_step=2
    )

    logger.log(
        time=61,
        state=state,
        arrived_this_step=1
    )

    # --------------------------------------------------
    # VERIFY FILE
    # --------------------------------------------------

    assert os.path.exists(output_file)

    with open(
        output_file,
        "r",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        reader = csv.DictReader(csv_file)

        rows = list(reader)

    # --------------------------------------------------
    # VERIFY ROW COUNT
    # --------------------------------------------------

    assert len(rows) == 2

    # --------------------------------------------------
    # VERIFY COLUMNS
    # --------------------------------------------------

    expected_columns = [
        "time",
        "north_queue",
        "south_queue",
        "east_queue",
        "west_queue",
        "north_waiting",
        "south_waiting",
        "east_waiting",
        "west_waiting",
        "signal_phase",
        "vehicle_count",
        "arrived_this_step"
    ]

    assert reader.fieldnames == expected_columns

    # --------------------------------------------------
    # VERIFY FIRST ROW
    # --------------------------------------------------

    assert rows[0]["time"] == "60"
    assert rows[0]["north_queue"] == "2"
    assert rows[0]["south_queue"] == "1"
    assert rows[0]["east_queue"] == "3"
    assert rows[0]["west_queue"] == "0"

    assert rows[0]["north_waiting"] == "4.0"
    assert rows[0]["south_waiting"] == "2.0"
    assert rows[0]["east_waiting"] == "6.0"
    assert rows[0]["west_waiting"] == "0.0"

    assert rows[0]["signal_phase"] == "0"
    assert rows[0]["vehicle_count"] == "7"
    assert rows[0]["arrived_this_step"] == "2"

    # --------------------------------------------------
    # VERIFY SECOND ROW
    # --------------------------------------------------

    assert rows[1]["time"] == "61"
    assert rows[1]["arrived_this_step"] == "1"

    print()
    print(f"CSV file: {output_file}")
    print(f"Rows written: {len(rows)}")
    print(f"Columns verified: {len(reader.fieldnames)}")

    print()
    print("=" * 100)
    print("✅ TRAFFIC LOGGER TEST PASSED")
    print("=" * 100)


if __name__ == "__main__":
    main()