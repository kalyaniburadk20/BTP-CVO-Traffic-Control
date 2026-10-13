import os
import sys
import csv
import traci


# Project root
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# Allow imports from project root
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


SCENARIOS = [
    "low",
    "medium",
    "high"
]


RAW_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "raw"
)


def calculate_throughput(scenario_name):

    config_file = os.path.join(
        PROJECT_ROOT,
        "sumo",
        "configs",
        f"{scenario_name}.sumocfg"
    )

    total_arrived = 0

    sumo_cmd = [
        "sumo",
        "-c",
        config_file
    ]

    print()
    print("=" * 60)
    print(f"Calculating throughput: {scenario_name.upper()}")
    print("=" * 60)

    traci.start(sumo_cmd)

    try:

        while traci.simulation.getMinExpectedNumber() > 0:

            traci.simulationStep()

            # Vehicles that reached destination in this step
            arrived_this_step = traci.simulation.getArrivedNumber()

            total_arrived += arrived_this_step

    finally:
        traci.close()

    print(
        f"Throughput: {total_arrived} vehicles"
    )

    return total_arrived


def main():

    results = []

    for scenario in SCENARIOS:

        throughput = calculate_throughput(
            scenario
        )

        results.append({
            "scenario": scenario.upper(),
            "throughput": throughput
        })

    # Save summary
    output_file = os.path.join(
        RAW_DIR,
        "throughput_summary.csv"
    )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "scenario",
                "throughput"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print("=" * 60)
    print("THROUGHPUT SUMMARY")
    print("=" * 60)

    for result in results:

        print(
            f"{result['scenario']:<10}"
            f"{result['throughput']} vehicles"
        )

    print("=" * 60)
    print()
    print(
        f"Saved to: {output_file}"
    )


if __name__ == "__main__":
    main()