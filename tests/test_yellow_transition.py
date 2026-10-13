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
    print("TEST — YELLOW TRANSITION")
    print("=" * 80)

    executor = SignalExecutor()

    try:

        # --------------------------------------------------
        # Initialize
        # --------------------------------------------------

        traci.simulationStep()

        current_time = traci.simulation.getTime()

        executor.initialize()

        print(
            f"Initial: "
            f"Time={current_time:.0f} | "
            f"SUMO Phase={traci.trafficlight.getPhase('J0')} | "
            f"Action={executor.current_action}"
        )

        # --------------------------------------------------
        # Wait for minimum green time
        # --------------------------------------------------

        for _ in range(10):
            traci.simulationStep()

        current_time = traci.simulation.getTime()

        print(
            f"Before switch: "
            f"Time={current_time:.0f} | "
            f"SUMO Phase={traci.trafficlight.getPhase('J0')} | "
            f"Action={executor.current_action}"
        )

        # --------------------------------------------------
        # Request opposite action
        # --------------------------------------------------

        result = executor.apply_action(
            "EW_GREEN"
        )

        current_phase = traci.trafficlight.getPhase("J0")

        print(
            f"Switch requested: "
            f"Time={current_time:.0f} | "
            f"SUMO Phase={current_phase} | "
            f"Executor Action={result} | "
            f"Pending={executor.pending_action}"
        )

        # --------------------------------------------------
        # Advance SUMO one step
        # --------------------------------------------------

        traci.simulationStep()

        current_time = traci.simulation.getTime()

        current_phase = traci.trafficlight.getPhase("J0")

        print(
            f"After 1 step: "
            f"Time={current_time:.0f} | "
            f"SUMO Phase={current_phase} | "
            f"Executor Action={executor.current_action} | "
            f"Yellow={executor.in_yellow}"
        )

        # --------------------------------------------------
        # Continue until yellow is finished
        # --------------------------------------------------

        for _ in range(5):

            traci.simulationStep()

            current_time = traci.simulation.getTime()

            current_phase = traci.trafficlight.getPhase("J0")

            print(
                f"Time={current_time:.0f} | "
                f"SUMO Phase={current_phase} | "
                f"Executor Action={executor.current_action} | "
                f"Yellow={executor.in_yellow}"
            )

            # Let executor detect completion
            if executor.in_yellow:
                executor.apply_action("EW_GREEN")

        print()
        print("=" * 80)
        print(
            f"Final SUMO phase: "
            f"{traci.trafficlight.getPhase('J0')}"
        )

        print(
            f"Final executor action: "
            f"{executor.current_action}"
        )

        print(
            f"Yellow active: "
            f"{executor.in_yellow}"
        )

    finally:

        traci.close()

        print()
        print("SUMO connection closed")


if __name__ == "__main__":
    main()