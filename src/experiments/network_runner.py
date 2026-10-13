import os
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


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

NETWORK_CONFIG_FILE = os.path.join(
    PROJECT_ROOT,
    "configs",
    "experiments",
    "network_master.yaml"
)


# --------------------------------------------------
# NETWORK INTERSECTIONS
# --------------------------------------------------

INTERSECTIONS = [
    "J1",
    "J2",
    "J3",
    "J4",
    "J5"
]


# --------------------------------------------------
# STRATEGIES
# --------------------------------------------------

STRATEGIES = [
    "always",
    "periodic",
    "state_change",
    "aoi",
    "cvo"
]


# --------------------------------------------------
# LOAD NETWORK CONFIG
# --------------------------------------------------

def load_network_config():

    if not os.path.exists(
        NETWORK_CONFIG_FILE
    ):
        raise FileNotFoundError(
            f"Network configuration not found: "
            f"{NETWORK_CONFIG_FILE}"
        )

    with open(
        NETWORK_CONFIG_FILE,
        "r",
        encoding="utf-8"
    ) as config_file:

        return yaml.safe_load(
            config_file
        )


# --------------------------------------------------
# SUMO CONFIG
# --------------------------------------------------

def get_network_sumo_config(scenario):

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
        f"{scenario}_network.sumocfg"
    )

    if not os.path.exists(
        config_file
    ):
        raise FileNotFoundError(
            f"SUMO config not found: "
            f"{config_file}"
        )

    return config_file


# --------------------------------------------------
# RUN ID
# --------------------------------------------------

def create_network_run_id(
    scenario,
    strategy,
    seed
):

    return (
        f"network__"
        f"{scenario}__"
        f"{strategy}__"
        f"seed{seed:02d}"
    )


# --------------------------------------------------
# RUN DIRECTORY
# --------------------------------------------------

def get_network_run_directory(run_id):

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
# BASELINE POLICY CREATION
# --------------------------------------------------

def create_network_baseline_policies(
    strategy,
    config
):
    """
    Create one independent baseline policy
    for every network intersection.
    """

    from src.acquisition.network_policy_factory import (
        create_network_policies
    )

    if strategy not in (
        "always",
        "periodic",
        "state_change",
        "aoi"
    ):
        raise ValueError(
            f"'{strategy}' is not a baseline strategy."
        )

    return create_network_policies(
        strategy=strategy,
        intersection_ids=INTERSECTIONS,
        config=config
    )

# --------------------------------------------------
# CVO COMPONENT CREATION
# --------------------------------------------------

def create_network_cvo_components(
    config
):
    """
    Create the independent CVO/control components
    for all network intersections.
    """

    from src.perception.belief import Belief

    from src.control.network_controller import (
        NetworkSignalController
    )

    from src.control.network_signal_executor import (
        NetworkSignalExecutor
    )

    from src.cvo.hypothetical import (
        HypotheticalObservationGenerator
    )

    from src.cvo.observation_cost import (
        ObservationCost
    )

    from src.acquisition.network_planner import (
        NetworkObservationPlanner
    )

    beliefs = {
        intersection_id: Belief()
        for intersection_id in INTERSECTIONS
    }

    controllers = {
        intersection_id:
            NetworkSignalController(
                intersection_id
            )
        for intersection_id in INTERSECTIONS
    }

    executors = {
        intersection_id:
            NetworkSignalExecutor(
                intersection_id
            )
        for intersection_id in INTERSECTIONS
    }

    hypothetical_generator = (
        HypotheticalObservationGenerator(
            east_queue_delta=config[
                "cvo"
            ][
                "hypothetical_east_queue_delta"
            ]
        )
    )

    cost_model = ObservationCost(
        compute_weight=config[
            "cvo"
        ][
            "cost_weights"
        ][
            "compute"
        ],
        communication_weight=config[
            "cvo"
        ][
            "cost_weights"
        ][
            "communication"
        ],
        latency_weight=config[
            "cvo"
        ][
            "cost_weights"
        ][
            "latency"
        ]
    )

    planner = NetworkObservationPlanner(
        observation_budget=config[
            "network"
        ][
            "observation_budget"
        ],
        max_belief_age=config[
            "network"
        ][
            "max_belief_age"
        ]
    )

    return {
        "beliefs": beliefs,
        "controllers": controllers,
        "executors": executors,
        "hypothetical_generator": (
            hypothetical_generator
        ),
        "cost_model": cost_model,
        "planner": planner
    }

# --------------------------------------------------
# NETWORK BUDGET MANAGER
# --------------------------------------------------

def create_network_budget_manager(
    config
):
    """
    Create the common network-wide observation
    budget manager.
    """

    from src.acquisition.network_budget_manager import (
        NetworkBudgetManager
    )

    return NetworkBudgetManager(
        observation_budget=config[
            "network"
        ][
            "observation_budget"
        ],
        intersection_ids=INTERSECTIONS
    )

# --------------------------------------------------
# NETWORK LOGGERS
# --------------------------------------------------

def create_network_loggers(
    run_directory
):
    """
    Create all network experiment loggers.
    """

    from src.experiments.network_traffic_logger import (
        NetworkTrafficLogger
    )

    from src.acquisition.network_acquisition_logger import (
        NetworkAcquisitionLogger
    )

    from src.cvo.network_cvo_logger import (
        NetworkCVOLogger
    )

    from src.cvo.network_decision_logger import (
        NetworkDecisionLogger
    )

    traffic_logger = NetworkTrafficLogger(
        os.path.join(
            run_directory,
            "traffic.csv"
        )
    )

    acquisition_logger = NetworkAcquisitionLogger(
        os.path.join(
            run_directory,
            "acquisition.csv"
        )
    )

    cvo_logger = NetworkCVOLogger(
        os.path.join(
            run_directory,
            "cvo.csv"
        )
    )

    decision_logger = NetworkDecisionLogger(
        os.path.join(
            run_directory,
            "decision.csv"
        )
    )

    return {
        "traffic": traffic_logger,
        "acquisition": acquisition_logger,
        "cvo": cvo_logger,
        "decision": decision_logger
    }

# --------------------------------------------------
# NETWORK RUN INITIALIZATION
# --------------------------------------------------

def initialize_network_run(
    scenario,
    strategy,
    seed
):
    """
    Initialize one network experiment run.

    This function prepares:
        - configuration
        - run directory
        - SUMO configuration
        - beliefs
        - controllers
        - executors
        - observation planner
        - budget manager
        - loggers
    """

    config = load_network_config()

    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    if scenario not in (
        "low",
        "medium",
        "high"
    ):
        raise ValueError(
            f"Unknown scenario: {scenario}"
        )

    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy: {strategy}"
        )

    if seed not in config["random"]["seeds"]:
        raise ValueError(
            f"Seed {seed} is not configured."
        )

    # --------------------------------------------------
    # RUN ID / DIRECTORY
    # --------------------------------------------------

    run_id = create_network_run_id(
        scenario=scenario,
        strategy=strategy,
        seed=seed
    )

    run_directory = get_network_run_directory(
        run_id
    )

    # --------------------------------------------------
    # SUMO
    # --------------------------------------------------

    sumo_config = get_network_sumo_config(
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
    # CVO / NETWORK COMPONENTS
    # --------------------------------------------------

    cvo_components = (
        create_network_cvo_components(
            config
        )
    )

    budget_manager = (
        create_network_budget_manager(
            config
        )
    )

    # --------------------------------------------------
    # BASELINE POLICIES
    # --------------------------------------------------

    baseline_policies = None

    if strategy != "cvo":

        baseline_policies = (
            create_network_baseline_policies(
                strategy=strategy,
                config=config
            )
        )

    # --------------------------------------------------
    # LOGGERS
    # --------------------------------------------------

    loggers = create_network_loggers(
        run_directory
    )

    return {
        "config": config,
        "run_id": run_id,
        "run_directory": run_directory,
        "sumo_cmd": sumo_cmd,

        "beliefs":
            cvo_components["beliefs"],

        "controllers":
            cvo_components["controllers"],

        "executors":
            cvo_components["executors"],

        "hypothetical_generator":
            cvo_components[
                "hypothetical_generator"
            ],

        "cost_model":
            cvo_components["cost_model"],

        "planner":
            cvo_components["planner"],

        "budget_manager":
            budget_manager,

        "baseline_policies":
            baseline_policies,

        "loggers":
            loggers
    }

# --------------------------------------------------
# START NETWORK SUMO
# --------------------------------------------------

def start_network_sumo(
    run_state
):
    """
    Start SUMO for one initialized network run
    and acquire the initial observation for all
    five intersections.
    """

    from src.perception.network_traffic_state import (
        get_intersection_traffic_state
    )

    from src.perception.observation import (
        Observation
    )

    traci.start(
        run_state["sumo_cmd"]
    )

    # First simulation step
    traci.simulationStep()

    current_time = traci.simulation.getTime()

    # --------------------------------------------------
    # Bootstrap all intersections
    # --------------------------------------------------

    for intersection_id in INTERSECTIONS:

        state = get_intersection_traffic_state(
            intersection_id
        )

        run_state["beliefs"][
            intersection_id
        ].update(
            Observation(
                traffic_state=state,
                timestamp=current_time
            )
        )

        # Initialize corresponding signal executor
        run_state["executors"][
            intersection_id
        ].initialize()

    return current_time

# --------------------------------------------------
# 10-STEP NETWORK PILOT
# --------------------------------------------------

def run_network_pilot(
    scenario,
    strategy,
    seed
):
    """
    Run a short 10-step pilot to validate the
    complete network acquisition/control loop.
    """

    from src.perception.network_traffic_state import (
        get_intersection_traffic_state
    )

    from src.perception.observation import (
        Observation
    )

    from src.cvo.cvo import (
        calculate_cvo,
        calculate_decision_margin
    )

    run_state = initialize_network_run(
        scenario=scenario,
        strategy=strategy,
        seed=seed
    )

    config = run_state["config"]

    try:

        start_network_sumo(
            run_state
        )

        print("=" * 80)
        print("NETWORK PILOT")
        print("=" * 80)
        print(
            f"Scenario: {scenario}"
        )
        print(
            f"Strategy: {strategy}"
        )
        print(
            f"Seed: {seed}"
        )
        print()

        for _ in range(20):

            traci.simulationStep()

            current_time = (
                traci.simulation.getTime()
            )

            cvo_values = {}
            cost_values = {}
            decision_margins = {}

            current_actions = {}
            hypothetical_actions = {}
            cost_results = {}
            # --------------------------------------------------
            # CVO VALUES
            # --------------------------------------------------

            if strategy == "cvo":

                for intersection_id in INTERSECTIONS:

                    belief = run_state[
                        "beliefs"
                    ][intersection_id]

                    hypothetical_belief = (
                        run_state[
                            "hypothetical_generator"
                        ].generate(
                            belief
                        )
                    )

                    controller = run_state[
                        "controllers"
                    ][intersection_id]

                    current_action = controller.choose_action(
                        belief
                    )

                    hypothetical_action = controller.choose_action(
                        hypothetical_belief
                    )

                    cvo = calculate_cvo(
                        current_belief=belief,
                        hypothetical_belief=hypothetical_belief,
                        controller=controller
                    )

                    decision_margin = calculate_decision_margin(
                        belief
                    )

                    cvo_values[intersection_id] = cvo

                    decision_margins[intersection_id] = decision_margin

                    current_actions[intersection_id] = current_action
                    hypothetical_actions[intersection_id] = hypothetical_action
                

                    run_state["loggers"]["cvo"].log(
                        time=current_time,
                        intersection_id=intersection_id,
                        current_belief=belief,
                        hypothetical_belief=hypothetical_belief,
                        current_action=current_action,
                        hypothetical_action=hypothetical_action,
                        decision_margin=decision_margin,
                        cvo=cvo,
                    )

                    cost_result = (
                        run_state[
                            "cost_model"
                        ].calculate(
                            compute_cost=40,
                            communication_cost=2,
                            latency_cost=80,
                            compute_reference=config[
                                "cvo"
                            ][
                                "reference_costs"
                            ][
                                "compute"
                            ],
                            communication_reference=config[
                                "cvo"
                            ][
                                "reference_costs"
                            ][
                                "communication"
                            ],
                            latency_reference=config[
                                "cvo"
                            ][
                                "reference_costs"
                            ][
                                "latency"
                            ]
                        )
                    )

                    cost_values[
                        intersection_id
                    ] = cost_result[
                        "total_cost"
                    ]

                    cost_results[intersection_id] = cost_result

                selected = run_state[
                    "planner"
                ].select_intersections(
                    cvo_values=cvo_values,
                    cost_values=cost_values,
                    decision_margins=decision_margins,
                    belief_ages={
                        i: run_state[
                            "beliefs"
                        ][i].get_age(
                            current_time
                        )
                        for i in INTERSECTIONS
                    }
                )

            else:

                # --------------------------------------------------
                # BASELINE REQUESTS
                # --------------------------------------------------

                requests = {}

                for intersection_id in INTERSECTIONS:

                    policy = run_state[
                        "baseline_policies"
                    ][intersection_id]

                    if strategy == "state_change":

                        true_state = (
                            get_intersection_traffic_state(
                                intersection_id
                            )
                        )


                        from src.perception.belief import (
                            Belief
                        )

                        temporary_belief = Belief(
                            state=true_state,
                            last_observation_time=current_time
                        )

                        requests[
                            intersection_id
                        ] = policy.should_acquire(
                            current_time=current_time,
                            belief=temporary_belief,
                            observation_history=[]
                        )

                    else:

                        requests[
                            intersection_id
                        ] = policy.should_acquire(
                            current_time=current_time,
                            belief=run_state[
                                "beliefs"
                            ][intersection_id],
                            observation_history=[]
                        )

                forced_intersections = [
                    intersection_id
                    for intersection_id in INTERSECTIONS
                    if (
                        run_state[
                            "beliefs"
                        ][intersection_id].get_age(
                            current_time
                        ) >= config[
                            "network"
                        ][
                            "max_belief_age"
                        ]
                    )
                ]

                selected = (
                    run_state[
                        "budget_manager"
                    ].select_requested(
                        requests=requests,
                        forced_intersections=forced_intersections
                    )
                )

            # --------------------------------------------------
            # ACQUIRE SELECTED OBSERVATIONS
            # --------------------------------------------------

            for intersection_id in selected:

                state = get_intersection_traffic_state(
                    intersection_id
                )

                run_state[
                    "beliefs"
                ][intersection_id].update(
                    Observation(
                        traffic_state=state,
                        timestamp=current_time
                    )
                )

                # State-change reference state must
                # only update after actual acquisition.
                if strategy == "state_change":

                    run_state[
                        "baseline_policies"
                    ][intersection_id].record_acquisition(
                        state
                    )

        # --------------------------------------------------
            # CONTROL ALL INTERSECTIONS
            # --------------------------------------------------

            for intersection_id in INTERSECTIONS:

                desired_action = (
                    run_state[
                        "controllers"
                    ][intersection_id].choose_action(
                        run_state[
                            "beliefs"
                        ][intersection_id]
                    )
                )

                actual_action = (
                    run_state[
                        "executors"
                    ][intersection_id].apply_action(
                        desired_action
                    )
                )

                run_state[
                    "loggers"
                ]["acquisition"].log(
                    time=current_time,
                    intersection_id=intersection_id,
                    policy=strategy,
                    decision=(
                        intersection_id
                        in selected
                    )
                )

                if strategy == "cvo":
                    run_state["loggers"]["decision"].log(
                        time=current_time,
                        intersection_id=intersection_id,
                        current_action=current_actions[
                            intersection_id
                        ],
                        hypothetical_action=hypothetical_actions[
                            intersection_id
                        ],
                        cvo=cvo_values[
                            intersection_id
                        ],
                        cost_result=cost_results[
                            intersection_id
                        ],
                        decision=(
                            intersection_id
                            in selected
                        )
                    )

                traffic_state = get_intersection_traffic_state(
                    intersection_id
                )

                run_state["loggers"]["traffic"].log(
                    time=current_time,
                    intersection_id=intersection_id,
                    state=traffic_state,
                    arrived_this_step=len(
                        traci.simulation.getArrivedIDList()
                    ),
                )

            print(
                f"t={current_time:.0f}",
                "| Selected=",
                selected,
                "| Ages=",
                {
                    i: round(
                        run_state[
                            "beliefs"
                        ][i].get_age(
                            current_time
                        ),
                        1
                    )
                    for i in INTERSECTIONS
                }
            )

        print()
        print("Network pilot completed.")

    finally:

        traci.close()
        print("SUMO connection closed.")

# --------------------------------------------------
# FULL NETWORK EXPERIMENT
# --------------------------------------------------

def run_network_experiment(
    scenario,
    strategy,
    seed
):
    """
    Run one complete network experiment
    using the configured simulation duration.
    """

    from src.perception.network_traffic_state import (
        get_intersection_traffic_state
    )

    from src.perception.observation import (
        Observation
    )

    from src.cvo.cvo import (
        calculate_cvo,
        calculate_decision_margin
    )

    config = load_network_config()

    run_state = initialize_network_run(
        scenario=scenario,
        strategy=strategy,
        seed=seed
    )

    config = run_state["config"]

    try:

        start_network_sumo(run_state)

        print("=" * 80)
        print("FULL NETWORK EXPERIMENT")
        print("=" * 80)

        print(
            f"Scenario: {scenario}"
        )

        print(
            f"Strategy: {strategy}"
        )

        print(
            f"Seed: {seed}"
        )

        print(
            f"Duration: "
            f"{config['simulation']['duration']} s"
        )

        print(
            f"Warm-up: "
            f"{config['simulation']['warmup']} s"
        )

        print(
            f"Measurement: "
            f"{config['simulation']['measurement_start']}"
            f" <= t < "
            f"{config['simulation']['measurement_end']}"
        )

        print()

        print(
            "Full experiment runner initialized."
        )

        for _ in range(
            int(config["simulation"]["duration"]) - 1
        ):

            traci.simulationStep()

            current_time = (
                traci.simulation.getTime()
            )

            cvo_values = {}
            cost_values = {}
            decision_margins = {}

            current_actions = {}
            hypothetical_actions = {}
            cost_results = {}

            # --------------------------------------------------
            # CVO VALUES
            # --------------------------------------------------

            if strategy == "cvo":

                for intersection_id in INTERSECTIONS:

                    belief = run_state[
                        "beliefs"
                    ][intersection_id]

                    hypothetical_belief = (
                        run_state[
                            "hypothetical_generator"
                        ].generate(
                            belief
                        )
                    )

                    controller = run_state[
                        "controllers"
                    ][intersection_id]

                    current_action = (
                        controller.choose_action(
                            belief
                        )
                    )

                    hypothetical_action = (
                        controller.choose_action(
                            hypothetical_belief
                        )
                    )

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

                    cvo_values[
                        intersection_id
                    ] = cvo

                    decision_margins[
                        intersection_id
                    ] = decision_margin

                    current_actions[
                        intersection_id
                    ] = current_action

                    hypothetical_actions[
                        intersection_id
                    ] = hypothetical_action

                    run_state[
                        "loggers"
                    ]["cvo"].log(
                        time=current_time,
                        intersection_id=intersection_id,
                        current_belief=belief,
                        hypothetical_belief=hypothetical_belief,
                        current_action=current_action,
                        hypothetical_action=hypothetical_action,
                        decision_margin=decision_margin,
                        cvo=cvo
                    )

                    cost_result = (
                        run_state[
                            "cost_model"
                        ].calculate(
                            compute_cost=40,
                            communication_cost=2,
                            latency_cost=80,
                            compute_reference=config[
                                "cvo"
                            ][
                                "reference_costs"
                            ][
                                "compute"
                            ],
                            communication_reference=config[
                                "cvo"
                            ][
                                "reference_costs"
                            ][
                                "communication"
                            ],
                            latency_reference=config[
                                "cvo"
                            ][
                                "reference_costs"
                            ][
                                "latency"
                            ]
                        )
                    )

                    cost_values[
                        intersection_id
                    ] = cost_result[
                        "total_cost"
                    ]

                    cost_results[
                        intersection_id
                    ] = cost_result

                selected = (
                    run_state[
                        "planner"
                    ].select_intersections(
                        cvo_values=cvo_values,
                        cost_values=cost_values,
                        decision_margins=decision_margins,
                        belief_ages={
                            i: run_state[
                                "beliefs"
                            ][i].get_age(
                                current_time
                            )
                            for i in INTERSECTIONS
                        }
                    )
                )

            else:

                # --------------------------------------------------
                # BASELINE REQUESTS
                # --------------------------------------------------

                requests = {}

                for intersection_id in INTERSECTIONS:

                    policy = run_state[
                        "baseline_policies"
                    ][intersection_id]

                    if strategy == "state_change":

                        true_state = (
                            get_intersection_traffic_state(
                                intersection_id
                            )
                        )

                        from src.perception.belief import (
                            Belief
                        )

                        temporary_belief = Belief(
                            state=true_state,
                            last_observation_time=current_time
                        )

                        requests[
                            intersection_id
                        ] = policy.should_acquire(
                            current_time=current_time,
                            belief=temporary_belief,
                            observation_history=[]
                        )

                    else:

                        requests[
                            intersection_id
                        ] = policy.should_acquire(
                            current_time=current_time,
                            belief=run_state[
                                "beliefs"
                            ][intersection_id],
                            observation_history=[]
                        )

                forced_intersections = [
                    intersection_id
                    for intersection_id in INTERSECTIONS
                    if (
                        run_state[
                            "beliefs"
                        ][intersection_id].get_age(
                            current_time
                        ) >= config[
                            "network"
                        ][
                            "max_belief_age"
                        ]
                    )
                ]

                selected = (
                    run_state[
                        "budget_manager"
                    ].select_requested(
                        requests=requests,
                        forced_intersections=forced_intersections
                    )
                )

            # --------------------------------------------------
            # ACQUIRE SELECTED OBSERVATIONS
            # --------------------------------------------------

            for intersection_id in selected:

                state = (
                    get_intersection_traffic_state(
                        intersection_id
                    )
                )

                run_state[
                    "beliefs"
                ][intersection_id].update(
                    Observation(
                        traffic_state=state,
                        timestamp=current_time
                    )
                )

                # State-change reference state must
                # only update after actual acquisition.
                if strategy == "state_change":

                    run_state[
                        "baseline_policies"
                    ][intersection_id].record_acquisition(
                        state
                    )

            # --------------------------------------------------
            # CONTROL ALL INTERSECTIONS
            # --------------------------------------------------

            for intersection_id in INTERSECTIONS:

                desired_action = (
                    run_state[
                        "controllers"
                    ][intersection_id].choose_action(
                        run_state[
                            "beliefs"
                        ][intersection_id]
                    )
                )

                actual_action = (
                    run_state[
                        "executors"
                    ][intersection_id].apply_action(
                        desired_action
                    )
                )

                run_state[
                    "loggers"
                ]["acquisition"].log(
                    time=current_time,
                    intersection_id=intersection_id,
                    policy=strategy,
                    decision=(
                        intersection_id in selected
                    )
                )

                if strategy == "cvo":

                    run_state[
                        "loggers"
                    ]["decision"].log(
                        time=current_time,
                        intersection_id=intersection_id,
                        current_action=current_actions[
                            intersection_id
                        ],
                        hypothetical_action=hypothetical_actions[
                            intersection_id
                        ],
                        cvo=cvo_values[
                            intersection_id
                        ],
                        cost_result=cost_results[
                            intersection_id
                        ],
                        decision=(
                            intersection_id in selected
                        )
                    )

                traffic_state = (
                    get_intersection_traffic_state(
                        intersection_id
                    )
                )

                run_state[
                    "loggers"
                ]["traffic"].log(
                    time=current_time,
                    intersection_id=intersection_id,
                    state=traffic_state,
                    arrived_this_step=len(
                        traci.simulation.getArrivedIDList()
                    )
                )

        print()
        print("Full network simulation completed.")

    finally:

        traci.close()
        print("SUMO connection closed.")