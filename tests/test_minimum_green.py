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


from src.control.signal_executor import SignalExecutor


# --------------------------------------------------
# SUMO CONFIG
# --------------------------------------------------

CONFIG_FILE = os.path.join(
    PROJECT_ROOT,
    "sumo",
    "configs",
    "low.sumocfg"
)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    sumo_cmd = [
        "sumo",
        "-c",
        CONFIG_FILE
    ]

    traci.start(sumo_cmd)

    print("Connected to SUMO")
    print()
    print("TEST — MINIMUM GREEN CONSTRAINT")
    print("=" * 80)

    executor = SignalExecutor()

    try:

        # --------------------------------------------------
        # Start simulation
        # --------------------------------------------------

        traci.simulationStep()

        current_time = traci.simulation.getTime()

        executor.initialize()

        print(
            f"Initial | "
            f"Time={current_time:.0f} | "
            f"Current Action={executor.current_action}"
        )

        # --------------------------------------------------
        # Try switching immediately
        # --------------------------------------------------

        result = executor.apply_action("EW_GREEN")

        elapsed = executor.get_elapsed_green_time(
            current_time
        )

        print(
            f"Time={current_time:.0f} | "
            f"Elapsed={elapsed:.1f}s | "
            f"Requested=EW_GREEN | "
            f"Actual={result}"
        )

        # --------------------------------------------------
        # Move to 5 seconds
        # --------------------------------------------------

        for _ in range(5):
            traci.simulationStep()

        current_time = traci.simulation.getTime()

        result = executor.apply_action("EW_GREEN")

        elapsed = executor.get_elapsed_green_time(
            current_time
        )

        print(
            f"Time={current_time:.0f} | "
            f"Elapsed={elapsed:.1f}s | "
            f"Requested=EW_GREEN | "
            f"Actual={result}"
        )

        # --------------------------------------------------
        # Move to 11 seconds after initialization
        # --------------------------------------------------

        for _ in range(6):
            traci.simulationStep()

        current_time = traci.simulation.getTime()

        result = executor.apply_action("EW_GREEN")

        elapsed = executor.get_elapsed_green_time(
            current_time
        )

        print(
            f"Time={current_time:.0f} | "
            f"Elapsed={elapsed:.1f}s | "
            f"Requested=EW_GREEN | "
            f"Actual={result}"
        )

        # --------------------------------------------------
        # Final SUMO phase
        # --------------------------------------------------

        final_phase = traci.trafficlight.getPhase("J0")

        print()
        print(
            f"Final SUMO phase: {final_phase}"
        )

    finally:

        traci.close()

        print()
        print("SUMO connection closed")


# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":
    main()