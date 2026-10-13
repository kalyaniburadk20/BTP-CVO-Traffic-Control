import os
import sys
import csv
import yaml
import traci


# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# --------------------------------------------------
# PERCEPTION
# --------------------------------------------------

from src.perception.traffic_state import (
    get_traffic_state
)

from src.perception.observation import (
    Observation
)

from src.perception.belief import (
    Belief
)


# --------------------------------------------------
# CONTROL
# --------------------------------------------------

from src.control.controller import (
    SignalController
)

from src.control.signal_executor import (
    SignalExecutor
)


# --------------------------------------------------
# BASELINE POLICIES
# --------------------------------------------------

from src.acquisition.always_acquire import (
    AlwaysAcquirePolicy
)

from src.acquisition.periodic import (
    PeriodicPolicy
)

from src.acquisition.state_change import (
    StateChangePolicy
)

from src.acquisition.aoi import (
    AoIPolicy
)


# --------------------------------------------------
# CVO
# --------------------------------------------------

from src.cvo.cvo import (
    calculate_cvo,
    calculate_decision_margin
)

from src.cvo.observation_cost import (
    ObservationCost
)

from src.cvo.hypothetical import (
    HypotheticalObservationGenerator
)

from src.cvo.decision_logger import (
    CVODecisionLogger
)

from src.cvo.cvo_logger import (
    CVOLogger
)

from src.acquisition.acquisition_policy import (
    CVOAcquisitionPolicy
)


# --------------------------------------------------
# EXPERIMENT LOGGER
# --------------------------------------------------

from src.experiments.traffic_logger import (
    TrafficLogger
)


# --------------------------------------------------
# ACQUISITION LOGGER
# --------------------------------------------------

from src.acquisition.acquisition_logger import (
    AcquisitionLogger
)


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

MASTER_CONFIG = os.path.join(
    PROJECT_ROOT,
    "configs",
    "experiments",
    "master.yaml"
)


# --------------------------------------------------
# LOAD CONFIG
# --------------------------------------------------

def load_master_config():

    if not os.path.exists(MASTER_CONFIG):

        raise FileNotFoundError(
            f"Master configuration not found: "
            f"{MASTER_CONFIG}"
        )

    with open(
        MASTER_CONFIG,
        "r",
        encoding="utf-8"
    ) as file:

        return yaml.safe_load(file)


# --------------------------------------------------
# SUMO CONFIG
# --------------------------------------------------

def get_sumo_config(scenario):

    allowed_scenarios = {
        "low",
        "medium",
        "high"
    }

    if scenario not in allowed_scenarios:

        raise ValueError(
            f"Unknown scenario: {scenario}"
        )

    config_file = os.path.join(
        PROJECT_ROOT,
        "sumo",
        "configs",
        f"{scenario}.sumocfg"
    )

    if not os.path.exists(config_file):

        raise FileNotFoundError(
            f"SUMO config not found: {config_file}"
        )

    return config_file


# --------------------------------------------------
# BASELINE POLICY SELECTION
# --------------------------------------------------

def get_baseline_policy(
    strategy,
    config
):

    if strategy == "always":

        return AlwaysAcquirePolicy()

    if strategy == "periodic":

        return PeriodicPolicy(
            interval=config[
                "acquisition"
            ]["periodic_interval"]
        )

    if strategy == "state_change":

        return StateChangePolicy(
            threshold=config[
                "acquisition"
            ]["state_change_threshold"]
        )

    if strategy == "aoi":

        return AoIPolicy(
            max_age=config[
                "acquisition"
            ]["aoi_max_age"]
        )

    raise ValueError(
        f"'{strategy}' is not a baseline strategy."
    )


# --------------------------------------------------
# RUN ID
# --------------------------------------------------

def create_run_id(
    scenario,
    strategy,
    seed
):

    return (
        f"{scenario}__"
        f"{strategy}__"
        f"seed{seed:02d}"
    )


# --------------------------------------------------
# RUN DIRECTORY
# --------------------------------------------------

def get_run_directory(run_id):

    run_directory = os.path.join(
        PROJECT_ROOT,
        "results",
        "raw",
        run_id
    )

    os.makedirs(
        run_directory,
        exist_ok=True
    )

    return run_directory


# --------------------------------------------------
# RUN EXPERIMENT
# --------------------------------------------------

def run_experiment(
    scenario,
    strategy,
    seed
):

    config = load_master_config()

    # --------------------------------------------------
    # Validate
    # --------------------------------------------------

    if scenario not in config["traffic"]["scenarios"]:

        raise ValueError(
            f"Scenario '{scenario}' is not in master.yaml."
        )

    if strategy not in config["strategies"]:

        raise ValueError(
            f"Strategy '{strategy}' is not in master.yaml."
        )

    if seed not in config["random"]["seeds"]:

        raise ValueError(
            f"Seed '{seed}' is not in master.yaml."
        )

    # --------------------------------------------------
    # Run ID
    # --------------------------------------------------

    run_id = create_run_id(
        scenario=scenario,
        strategy=strategy,
        seed=seed
    )

    run_directory = get_run_directory(
        run_id
    )

    # --------------------------------------------------
    # Output files
    # --------------------------------------------------

    traffic_output = os.path.join(
        run_directory,
        "traffic.csv"
    )

    acquisition_output = os.path.join(
        run_directory,
        "acquisition.csv"
    )

    cvo_output = os.path.join(
        run_directory,
        "cvo.csv"
    )

    decision_output = os.path.join(
        run_directory,
        "decision.csv"
    )

    # --------------------------------------------------
    # SUMO
    # --------------------------------------------------

    sumo_config = get_sumo_config(
        scenario
    )

    sumo_cmd = [
        "sumo",
        "-c",
        sumo_config,
        "--seed",
        str(seed)
    ]

    # --------------------------------------------------
    # Common modules
    # --------------------------------------------------

    belief = Belief()

    controller = SignalController()

    executor = SignalExecutor()

    traffic_logger = TrafficLogger(
        traffic_output
    )

    acquisition_logger = AcquisitionLogger(
        acquisition_output
    )

    # --------------------------------------------------
    # Strategy-specific modules
    # --------------------------------------------------

    baseline_policy = None
    cvo_policy = None
    hypothetical_generator = None
    cost_model = None
    cvo_logger = None
    decision_logger = None

    if strategy == "cvo":

        cvo_policy = CVOAcquisitionPolicy()

        hypothetical_generator = (
            HypotheticalObservationGenerator(
                east_queue_delta=5
            )
        )

        cost_model = ObservationCost(
    compute_weight=config["cvo"]["cost_weights"]["compute"],
    communication_weight=config["cvo"]["cost_weights"]["communication"],
    latency_weight=config["cvo"]["cost_weights"]["latency"]
)

        cvo_logger = CVOLogger(
            cvo_output
        )

        decision_logger = CVODecisionLogger(
            decision_output
        )

    else:

        baseline_policy = get_baseline_policy(
            strategy=strategy,
            config=config
        )

    # --------------------------------------------------
    # Start SUMO
    # --------------------------------------------------

    traci.start(
        sumo_cmd
    )

    print("=" * 100)
    print("PHASE 8 — EXPERIMENT")
    print("=" * 100)

    print(
        f"Run ID:       {run_id}"
    )

    print(
        f"Scenario:     {scenario}"
    )

    print(
        f"Strategy:     {strategy}"
    )

    print(
        f"Seed:         {seed}"
    )

    print(
        f"Duration:     "
        f"{config['simulation']['duration']} s"
    )

    print(
        f"Warm-up:      "
        f"{config['simulation']['warmup']} s"
    )

    print()

    # --------------------------------------------------
    # Initialize signal executor
    # --------------------------------------------------

    traci.simulationStep()

    executor.initialize()

    print(
        f"Initial simulation time: "
        f"{traci.simulation.getTime():.0f}"
    )

    available_count = 0
    acquired_count = 0
    total_arrived = 0

    # --------------------------------------------------
    # MEASUREMENT WINDOW
    # --------------------------------------------------
    # We keep warm-up data in the raw logs, but evaluation
    # metrics are calculated only on [measurement_start,
    # measurement_end).
    measurement_start = config["simulation"]["measurement_start"]
    measurement_end = config["simulation"]["measurement_end"]

    measurement_available_count = 0
    measurement_acquired_count = 0
    measurement_arrived = 0
    measurement_queue_sum = 0.0
    measurement_waiting_sum = 0.0
    measurement_steps = 0

    observation_history = []

    try:

        while (
            traci.simulation.getTime()
            < config["simulation"]["duration"]
        ):

            traci.simulationStep()

            current_time = traci.simulation.getTime()

            # --------------------------------------------------
            # TRUE CURRENT TRAFFIC STATE
            # --------------------------------------------------

            true_state = get_traffic_state()

            arrived_this_step = (
                traci.simulation.getArrivedNumber()
            )

            total_arrived += arrived_this_step
            available_count += 1

            # Evaluation uses only the configured measurement window.
            # End time is exclusive, giving 540 measurement steps
            # for the current 60-600 configuration.
            in_measurement_window = (
                measurement_start <= current_time < measurement_end
            )

            if in_measurement_window:
                measurement_available_count += 1
                measurement_arrived += arrived_this_step
                measurement_steps += 1

                measurement_queue_sum += (
                    true_state.north_queue
                    + true_state.south_queue
                    + true_state.east_queue
                    + true_state.west_queue
                )

                measurement_waiting_sum += (
                    true_state.north_waiting
                    + true_state.south_waiting
                    + true_state.east_waiting
                    + true_state.west_waiting
                )

            # ==================================================
            # CVO STRATEGY
            # ==================================================

            if strategy == "cvo":

                # --------------------------------------------------
                # Bootstrap
                #
                # At the first opportunity there is no current
                # belief yet. One real observation is therefore
                # required to initialize the belief.
                # --------------------------------------------------

                if belief.state is None:

                    should_acquire = True

                    current_action = None
                    hypothetical_action = None
                    cvo = None
                    decision_margin = None
                    cost_result = None

                    acquired_count += 1

                    if in_measurement_window:
                        measurement_acquired_count += 1

                    observation = Observation(
                        traffic_state=true_state,
                        timestamp=current_time
                    )

                    belief.update(
                        observation
                    )

                    observation_history.append(
                        observation
                    )

                    desired_action = (
                        controller.choose_action(
                            belief
                        )
                    )

                    actual_action = (
                        executor.apply_action(
                            desired_action
                        )
                    )

                    acquisition_logger.log(
                        time=current_time,
                        policy=strategy,
                        decision=True
                    )

                    print(
                        f"Time={current_time:.0f} | "
                        f"Bootstrap=ACQUIRE | "
                        f"Belief initialized | "
                        f"Action={actual_action}"
                    )

                else:

                    # --------------------------------------------------
                    # Current action from EXISTING belief
                    # --------------------------------------------------

                    current_action = (
                        controller.choose_action(
                            belief
                        )
                    )

                    # --------------------------------------------------
                    # Hypothetical belief
                    #
                    # No future SUMO state is used.
                    # --------------------------------------------------

                    hypothetical_belief = (
                        hypothetical_generator.generate(
                            belief
                        )
                    )

                    # --------------------------------------------------
                    # Hypothetical action
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
                        current_belief=belief,
                        hypothetical_belief=hypothetical_belief,
                        controller=controller
                    )

                    decision_margin = (
                        calculate_decision_margin(
                            belief
                        )
                    )

                    # --------------------------------------------------
                    # Observation cost
                    # --------------------------------------------------

                    cost_result = cost_model.calculate(
                        compute_cost=40,
                        communication_cost=2,
                        latency_cost=80,
                        compute_reference=100,
                        communication_reference=5,
                        latency_reference=200
                    )

                    # --------------------------------------------------
                    # CVO > Cost
                    # --------------------------------------------------

                    should_acquire = (
                        cvo_policy.should_acquire(
                            cvo=cvo,
                            total_cost=cost_result[
                                "total_cost"
                            ]
                        )
                    )

                    if should_acquire:

                        acquired_count += 1

                        if in_measurement_window:
                            measurement_acquired_count += 1

                        # --------------------------------------------------
                        # Acquire actual current observation
                        # --------------------------------------------------

                        observation = Observation(
                            traffic_state=true_state,
                            timestamp=current_time
                        )

                        belief.update(
                            observation
                        )

                        observation_history.append(
                            observation
                        )

                    # --------------------------------------------------
                    # Log acquisition
                    # --------------------------------------------------

                    acquisition_logger.log(
                        time=current_time,
                        policy=strategy,
                        decision=should_acquire
                    )

                    # --------------------------------------------------
                    # Log CVO
                    # --------------------------------------------------

                    current_state = belief.state

                    current_belief_text = (
                        f"NS="
                        f"{current_state.north_queue + current_state.south_queue},"
                        f"EW="
                        f"{current_state.east_queue + current_state.west_queue}"
                    )

                    hypothetical_state = (
                        hypothetical_belief.state
                    )

                    hypothetical_belief_text = (
                        f"NS="
                        f"{hypothetical_state.north_queue + hypothetical_state.south_queue},"
                        f"EW="
                        f"{hypothetical_state.east_queue + hypothetical_state.west_queue}"
                    )

                    cvo_logger.log(
                        time=current_time,
                        current_belief=current_belief_text,
                        hypothetical_belief=hypothetical_belief_text,
                        current_action=current_action,
                        hypothetical_action=hypothetical_action,
                        decision_margin=decision_margin,
                        cvo=cvo
                    )

                    # --------------------------------------------------
                    # Log complete decision
                    # --------------------------------------------------

                    decision_logger.log(
                        time=current_time,
                        current_action=current_action,
                        hypothetical_action=hypothetical_action,
                        cvo=cvo,
                        cost_result=cost_result,
                        decision=should_acquire
                    )

                    # --------------------------------------------------
                    # Control continues whether ACQUIRE or SKIP
                    # --------------------------------------------------

                    if belief.state is not None:

                        desired_action = (
                            controller.choose_action(
                                belief
                            )
                        )

                        actual_action = (
                            executor.apply_action(
                                desired_action
                            )
                        )

                    else:

                        desired_action = None
                        actual_action = (
                            executor.current_action
                        )

                    print(
                        f"Time={current_time:.0f} | "
                        f"CVO={cvo:.1f} | "
                        f"Cost={cost_result['total_cost']:.2f} | "
                        f"Decision="
                        f"{'ACQUIRE' if should_acquire else 'SKIP'} | "
                        f"Current={current_action} | "
                        f"Hypothetical={hypothetical_action} | "
                        f"Actual={actual_action}"
                    )

            # ==================================================
            # BASELINE STRATEGIES
            # ==================================================

            else:

                should_acquire = (
                    baseline_policy.should_acquire(
                        current_time=current_time,
                        belief=belief,
                        observation_history=observation_history
                    )
                )

                if should_acquire:

                    acquired_count += 1

                    if in_measurement_window:
                        measurement_acquired_count += 1

                    observation = Observation(
                        traffic_state=true_state,
                        timestamp=current_time
                    )

                    belief.update(
                        observation
                    )

                    observation_history.append(
                        observation
                    )

                acquisition_logger.log(
                    time=current_time,
                    policy=strategy,
                    decision=should_acquire
                )

                if belief.state is not None:

                    desired_action = (
                        controller.choose_action(
                            belief
                        )
                    )

                    actual_action = (
                        executor.apply_action(
                            desired_action
                        )
                    )

                else:

                    desired_action = None
                    actual_action = (
                        executor.current_action
                    )

            # --------------------------------------------------
            # RAW TRAFFIC LOG
            # --------------------------------------------------

            traffic_logger.log(
                time=current_time,
                state=true_state,
                arrived_this_step=arrived_this_step
            )

        # --------------------------------------------------
        # SUMMARY
        # --------------------------------------------------

        acquisition_rate = (
            measurement_acquired_count / measurement_available_count
            if measurement_available_count > 0
            else 0.0
        )

        average_queue = (
            measurement_queue_sum / measurement_steps
            if measurement_steps > 0
            else 0.0
        )

        average_waiting = (
            measurement_waiting_sum / measurement_steps
            if measurement_steps > 0
            else 0.0
        )

        # Save one-row evaluation metrics for this run.
        metrics_output = os.path.join(
            run_directory,
            "evaluation_metrics.csv"
        )

        with open(
            metrics_output,
            "w",
            newline="",
            encoding="utf-8"
        ) as metrics_file:

            writer = csv.DictWriter(
                metrics_file,
                fieldnames=[
                    "run_id",
                    "scenario",
                    "strategy",
                    "seed",
                    "measurement_start",
                    "measurement_end",
                    "measurement_steps",
                    "available_observations",
                    "acquired_observations",
                    "acquisition_rate",
                    "average_queue_length",
                    "average_waiting_time",
                    "throughput"
                ]
            )

            writer.writeheader()

            writer.writerow({
                "run_id": run_id,
                "scenario": scenario,
                "strategy": strategy,
                "seed": seed,
                "measurement_start": measurement_start,
                "measurement_end": measurement_end,
                "measurement_steps": measurement_steps,
                "available_observations": measurement_available_count,
                "acquired_observations": measurement_acquired_count,
                "acquisition_rate": acquisition_rate,
                "average_queue_length": average_queue,
                "average_waiting_time": average_waiting,
                "throughput": measurement_arrived
            })

        print()
        print("-" * 100)

        print(
            f"Raw observations: "
            f"{available_count}"
        )

        print(
            f"Raw acquisitions: "
            f"{acquired_count}"
        )

        print(
            f"Measurement window: "
            f"{measurement_start} <= t < {measurement_end}"
        )

        print(
            f"Measurement steps: "
            f"{measurement_steps}"
        )

        print(
            f"Measurement observations: "
            f"{measurement_available_count}"
        )

        print(
            f"Measurement acquisitions: "
            f"{measurement_acquired_count}"
        )

        print(
            f"Measurement acquisition rate: "
            f"{acquisition_rate * 100:.1f}%"
        )

        print(
            f"Measurement average queue: "
            f"{average_queue:.2f}"
        )

        print(
            f"Measurement average waiting: "
            f"{average_waiting:.2f} s"
        )

        print(
            f"Measurement throughput: "
            f"{measurement_arrived} vehicles"
        )

        print()

        print(
            f"Traffic CSV: "
            f"{traffic_output}"
        )

        print(
            f"Acquisition CSV: "
            f"{acquisition_output}"
        )

        print(
            f"Evaluation metrics CSV: "
            f"{metrics_output}"
        )

        if strategy == "cvo":

            print(
                f"CVO CSV: "
                f"{cvo_output}"
            )

            print(
                f"Decision CSV: "
                f"{decision_output}"
            )

        print("=" * 100)

    finally:

        traci.close()

        print(
            "SUMO connection closed."
        )


# --------------------------------------------------
# COMMAND-LINE ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Run Phase 8 traffic-control experiment."
    )

    parser.add_argument(
        "--scenario",
        required=True,
        choices=["low", "medium", "high"],
        help="Traffic scenario."
    )

    parser.add_argument(
        "--strategy",
        required=True,
        choices=["always", "periodic", "state_change", "aoi", "cvo"],
        help="Observation acquisition strategy."
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed."
    )

    args = parser.parse_args()

    run_experiment(
        scenario=args.scenario,
        strategy=args.strategy,
        seed=args.seed
    )