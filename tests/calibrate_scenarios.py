import os
import csv
import statistics


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RAW_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "raw"
)


SCENARIOS = [
    "low",
    "medium",
    "high"
]


def analyze_scenario(scenario_name):

    file_path = os.path.join(
        RAW_DIR,
        f"{scenario_name}_traffic_raw.csv"
    )

    queue_values = []
    waiting_values = []
    vehicle_values = []

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as csv_file:

        reader = csv.DictReader(csv_file)

        for row in reader:

            # Total queue at this timestep
            total_queue = (
                int(row["north_queue"])
                + int(row["south_queue"])
                + int(row["east_queue"])
                + int(row["west_queue"])
            )

            # Total waiting at this timestep
            total_waiting = (
                float(row["north_waiting"])
                + float(row["south_waiting"])
                + float(row["east_waiting"])
                + float(row["west_waiting"])
            )

            vehicle_count = int(
                row["vehicle_count"]
            )

            queue_values.append(total_queue)
            waiting_values.append(total_waiting)
            vehicle_values.append(vehicle_count)

    average_queue = statistics.mean(queue_values)
    average_waiting = statistics.mean(waiting_values)

    maximum_queue = max(queue_values)
    maximum_waiting = max(waiting_values)

    average_vehicle_count = statistics.mean(
        vehicle_values
    )

    return {
        "scenario": scenario_name.upper(),
        "average_queue": average_queue,
        "average_waiting": average_waiting,
        "maximum_queue": maximum_queue,
        "maximum_waiting": maximum_waiting,
        "average_vehicle_count": average_vehicle_count
    }


def main():

    results = []

    for scenario in SCENARIOS:

        result = analyze_scenario(scenario)
        results.append(result)

    print()
    print("=" * 90)
    print("TRAFFIC SCENARIO CALIBRATION")
    print("=" * 90)

    print(
        f"{'Scenario':<12}"
        f"{'Avg Queue':<15}"
        f"{'Avg Waiting':<18}"
        f"{'Max Queue':<15}"
        f"{'Max Waiting':<18}"
        f"{'Avg Vehicles':<15}"
    )

    print("-" * 90)

    for result in results:

        print(
            f"{result['scenario']:<12}"
            f"{result['average_queue']:<15.2f}"
            f"{result['average_waiting']:<18.2f}"
            f"{result['maximum_queue']:<15}"
            f"{result['maximum_waiting']:<18.2f}"
            f"{result['average_vehicle_count']:<15.2f}"
        )

    print("=" * 90)


if __name__ == "__main__":
    main()