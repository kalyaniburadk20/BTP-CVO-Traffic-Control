import numpy as np
import traci

from types import SimpleNamespace

from src.control.network_controller import NetworkSignalController
from src.control.network_signal_executor import NetworkSignalExecutor
from src.control.network_signal_config import get_signal_config
from src.experiments.network_runner import (
    INTERSECTIONS,
    get_network_sumo_config,
    load_network_config,
)
from src.perception.network_traffic_state import (
    get_intersection_traffic_state,
)


class SingleIntersectionSumoEnv:
    """
    SUMO environment for training a single DQN agent.

    The learning agent controls one intersection.
    All remaining intersections use the existing
    queue-pressure-based controller.

    Actions:
        0 -> NS_GREEN
        1 -> EW_GREEN

    The existing NetworkSignalExecutor enforces minimum
    green time and yellow transitions.
    """

    ACTIONS = {
        0: "NS_GREEN",
        1: "EW_GREEN",
    }

    def __init__(
        self,
        scenario="low",
        seed=1,
        controlled_intersection="J1",
        reward_wait_weight=0.001,
        reward_queue_weight=0.10,
        switching_penalty=0.5,
    ):
        if scenario not in ("low", "medium", "high"):
            raise ValueError(f"Unknown scenario: {scenario}")

        if controlled_intersection not in INTERSECTIONS:
            raise ValueError(
                f"Unknown intersection: {controlled_intersection}"
            )

        self.config = load_network_config()
        self.scenario = scenario
        self.seed = int(seed)
        self.controlled_intersection = controlled_intersection

        self.duration = float(
            self.config["simulation"]["duration"]
        )
        self.warmup = float(
            self.config["simulation"]["warmup"]
        )

        self.reward_wait_weight = float(reward_wait_weight)
        self.reward_queue_weight = float(reward_queue_weight)
        self.switching_penalty = float(switching_penalty)

        self.controllers = {
            intersection_id: NetworkSignalController(intersection_id)
            for intersection_id in INTERSECTIONS
        }

        self.executors = {
            intersection_id: NetworkSignalExecutor(intersection_id)
            for intersection_id in INTERSECTIONS
        }

        self.started = False
        self.episode_steps = 0
        self.total_reward = 0.0
        self.episode_arrivals = 0
        self.last_state = None

        self.sumo_config = get_network_sumo_config(scenario)

        # Observation:
        # 4 normalized queues
        # 4 normalized waiting-time values
        # 2 current-action indicators
        # 1 normalized green duration
        # 1 transition indicator
        self.observation_size = 12
        self.action_size = 2

        get_signal_config(controlled_intersection)

    def _get_background_action(self, intersection_id):
        state = get_intersection_traffic_state(intersection_id)

        belief = SimpleNamespace(state=state)

        return self.controllers[
            intersection_id
        ].choose_action(belief)

    def _apply_background_controllers(self):
        """
        Apply existing pressure-based control to every
        intersection except the RL-controlled intersection.
        """

        for intersection_id in INTERSECTIONS:
            if intersection_id == self.controlled_intersection:
                continue

            desired_action = self._get_background_action(
                intersection_id
            )

            self.executors[
                intersection_id
            ].apply_action(desired_action)

    def _warmup_network(self):
        """
        Run the configured warm-up period with the
        existing pressure-based controller at all intersections.
        """

        while traci.simulation.getTime() < self.warmup:
            for intersection_id in INTERSECTIONS:
                desired_action = self._get_background_action(
                    intersection_id
                )

                self.executors[
                    intersection_id
                ].apply_action(desired_action)

            traci.simulationStep()

            if traci.simulation.getMinExpectedNumber() == 0:
                break

    def _synchronize_controlled_executor(self):
        """
        Synchronize executor state with SUMO after a simulation step.

        The executor normally detects transition completion on its
        next apply_action call. The RL environment synchronizes it
        before returning an observation to avoid a one-step lag.
        """

        executor = self.executors[self.controlled_intersection]

        if not executor.in_transition:
            return

        phase = traci.trafficlight.getPhase(
            self.controlled_intersection
        )
        config = get_signal_config(
            self.controlled_intersection
        )

        if phase == config["NS_GREEN_PHASE"]:
            action = "NS_GREEN"
        elif phase == config["EW_GREEN_PHASE"]:
            action = "EW_GREEN"
        else:
            return

        executor.current_action = action
        executor.phase_start_time = traci.simulation.getTime()
        executor.pending_action = None
        executor.in_transition = False

    def _get_observation(self):
        state = get_intersection_traffic_state(
            self.controlled_intersection
        )

        executor = self.executors[
            self.controlled_intersection
        ]

        current_time = traci.simulation.getTime()

        queues = [
            state.north_queue,
            state.south_queue,
            state.east_queue,
            state.west_queue,
        ]

        waiting = [
            state.north_waiting,
            state.south_waiting,
            state.east_waiting,
            state.west_waiting,
        ]

        current_action = executor.current_action

        action_features = [
            float(current_action == "NS_GREEN"),
            float(current_action == "EW_GREEN"),
        ]

        elapsed_green = executor.get_elapsed_green_time(
            current_time
        )

        observation = np.asarray(
            [
                *[
                    np.clip(float(value) / 30.0, 0.0, 1.0)
                    for value in queues
                ],
                *[
                    np.clip(float(value) / 300.0, 0.0, 1.0)
                    for value in waiting
                ],
                *action_features,
                np.clip(float(elapsed_green) / 60.0, 0.0, 1.0),
                float(executor.in_transition),
            ],
            dtype=np.float32,
        )

        if observation.shape != (self.observation_size,):
            raise RuntimeError(
                f"Invalid observation shape: {observation.shape}"
            )

        if not np.isfinite(observation).all():
            raise ValueError("Non-finite RL observation detected.")

        self.last_state = state
        return observation

    def reset(self, seed=None):
        """
        Start a fresh SUMO episode and return:
            observation, info
        """

        if self.started:
            self.close()

        if seed is not None:
            self.seed = int(seed)

        self.episode_steps = 0
        self.total_reward = 0.0
        self.episode_arrivals = 0
        self.last_state = None

        sumo_command = [
            "sumo",
            "-c",
            self.sumo_config,
            "--seed",
            str(self.seed),
        ]

        traci.start(sumo_command)
        self.started = True

        try:
            # Initialize SUMO and signal executors.
            traci.simulationStep()

            for intersection_id in INTERSECTIONS:
                self.executors[
                    intersection_id
                ].initialize()

            # Warm-up is controlled using the existing
            # pressure-based policy at all intersections.
            self._warmup_network()

            if (
                traci.simulation.getMinExpectedNumber() == 0
                or traci.simulation.getTime() >= self.duration
            ):
                raise RuntimeError(
                    "SUMO episode ended during warm-up."
                )

            observation = self._get_observation()

            info = {
                "scenario": self.scenario,
                "seed": self.seed,
                "controlled_intersection": self.controlled_intersection,
                "simulation_time": traci.simulation.getTime(),
                "warmup_duration": self.warmup,
                "observation_size": self.observation_size,
                "action_size": self.action_size,
            }

            return observation, info

        except Exception:
            self.close()
            raise

    def step(self, action):
        """
        Execute one RL action and advance the SUMO simulation.

        Returns:
            next_observation, reward, done, info
        """

        if not self.started:
            raise RuntimeError("Call reset() before step().")

        if action not in self.ACTIONS:
            raise ValueError(
                f"Invalid action {action}. Expected 0 or 1."
            )

        if traci.simulation.getTime() >= self.duration:
            raise RuntimeError("Episode duration already reached.")

        desired_action = self.ACTIONS[action]

        controlled_executor = self.executors[
            self.controlled_intersection
        ]

        was_in_transition = controlled_executor.in_transition

        controlled_executor.apply_action(desired_action)

        # A switching penalty is applied only when a real
        # yellow transition starts, not when an action is
        # merely requested or rejected due to minimum green.
        switch_started = (
            not was_in_transition
            and controlled_executor.in_transition
        )

        # Other intersections continue using the original
        # pressure-based controller.
        self._apply_background_controllers()

        # Advance the complete traffic network by one timestep.
        traci.simulationStep()

        # Synchronize transition completion before exposing
        # the next state to the RL agent.
        self._synchronize_controlled_executor()

        next_observation = self._get_observation()
        state = self.last_state

        total_queue = float(
            state.north_queue
            + state.south_queue
            + state.east_queue
            + state.west_queue
        )

        total_waiting = float(
            state.north_waiting
            + state.south_waiting
            + state.east_waiting
            + state.west_waiting
        )

        reward = -(
            self.reward_queue_weight * total_queue
            + self.reward_wait_weight * total_waiting
            + (
                self.switching_penalty
                if switch_started else 0.0
            )
        )

        self.episode_steps += 1
        self.total_reward += reward

        arrived_this_step = len(
            traci.simulation.getArrivedIDList()
        )
        self.episode_arrivals += arrived_this_step

        current_time = traci.simulation.getTime()

        terminated = (
            current_time >= self.duration
            or traci.simulation.getMinExpectedNumber() == 0
        )

        info = {
            "scenario": self.scenario,
            "seed": self.seed,
            "simulation_time": current_time,
            "controlled_intersection": self.controlled_intersection,
            "desired_action": desired_action,
            "active_action": controlled_executor.current_action,
            "transition_active": controlled_executor.in_transition,
            "switch_started": switch_started,
            "total_queue": total_queue,
            "total_waiting": total_waiting,
            "reward": float(reward),
            "episode_steps": self.episode_steps,
            "episode_reward": self.total_reward,
            "arrived_this_step": arrived_this_step,
            "episode_arrivals": self.episode_arrivals,
        }

        return next_observation, float(reward), bool(terminated), info

    def close(self):
        """
        Close the SUMO connection safely.
        """

        if self.started:
            try:
                traci.close()
            finally:
                self.started = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
