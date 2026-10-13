import traci

from src.control.network_signal_config import (
    get_signal_config
)


class NetworkSignalExecutor:
    """
    Executes signal actions for one intersection
    in the multi-intersection network.
    """

    MIN_GREEN_TIME = 10.0

    def __init__(self, intersection_id):

        self.intersection_id = intersection_id

        self.current_action = None
        self.phase_start_time = None
        self.pending_action = None
        self.in_transition = False

        get_signal_config(
            self.intersection_id
        )

    # --------------------------------------------------
    # INITIALIZATION
    # --------------------------------------------------

    def initialize(self):

        current_phase = traci.trafficlight.getPhase(
            self.intersection_id
        )

        current_time = traci.simulation.getTime()

        self.current_action = (
            self.phase_to_action(
                current_phase
            )
        )

        self.phase_start_time = current_time

        self.pending_action = None
        self.in_transition = False

    # --------------------------------------------------
    # PHASE -> ACTION
    # --------------------------------------------------

    def phase_to_action(self, phase):

        config = get_signal_config(
            self.intersection_id
        )

        if phase == config["NS_GREEN_PHASE"]:
            return "NS_GREEN"

        if phase == config["EW_GREEN_PHASE"]:
            return "EW_GREEN"

        # Yellow / intermediate phases belong
        # to the transition toward the next green.
        if self.intersection_id in ("J1", "J3"):

            if phase in (1, 2, 3):
                return "EW_GREEN"

        if phase == config["NS_YELLOW_PHASE"]:
            return "NS_GREEN"

        if phase == config["EW_YELLOW_PHASE"]:
            return "EW_GREEN"

        raise ValueError(
            f"Unknown phase {phase} for "
            f"{self.intersection_id}"
        )

    # --------------------------------------------------
    # ELAPSED TIME
    # --------------------------------------------------

    def get_elapsed_green_time(self, current_time):

        if self.phase_start_time is None:
            return 0.0

        return (
            current_time
            - self.phase_start_time
        )

    # --------------------------------------------------
    # APPLY ACTION
    # --------------------------------------------------

    def apply_action(self, desired_action):

        if desired_action not in (
            "NS_GREEN",
            "EW_GREEN"
        ):
            raise ValueError(
                f"Unknown desired action: "
                f"{desired_action}"
            )

        current_time = traci.simulation.getTime()

        if self.current_action is None:
            self.initialize()

        config = get_signal_config(
            self.intersection_id
        )

        current_phase = traci.trafficlight.getPhase(
            self.intersection_id
        )

        # --------------------------------------------------
        # CASE 1: Waiting for requested transition
        # --------------------------------------------------

        if self.in_transition:

            if current_phase == config[
                "NS_GREEN_PHASE"
            ]:

                self.current_action = "NS_GREEN"
                self.phase_start_time = current_time
                self.pending_action = None
                self.in_transition = False

                return self.current_action

            if current_phase == config[
                "EW_GREEN_PHASE"
            ]:

                self.current_action = "EW_GREEN"
                self.phase_start_time = current_time
                self.pending_action = None
                self.in_transition = False

                return self.current_action

            return self.current_action

        # --------------------------------------------------
        # CASE 2: Desired action already active
        # --------------------------------------------------

        if desired_action == self.current_action:
            return self.current_action

        # --------------------------------------------------
        # CASE 3: Minimum green not completed
        # --------------------------------------------------

        elapsed_time = (
            self.get_elapsed_green_time(
                current_time
            )
        )

        if elapsed_time < self.MIN_GREEN_TIME:
            return self.current_action

        # --------------------------------------------------
        # CASE 4: Start transition
        # --------------------------------------------------

        self.pending_action = desired_action
        self.in_transition = True

        if self.current_action == "NS_GREEN":

            traci.trafficlight.setPhase(
                self.intersection_id,
                config["NS_YELLOW_PHASE"]
            )

        elif self.current_action == "EW_GREEN":

            traci.trafficlight.setPhase(
                self.intersection_id,
                config["EW_YELLOW_PHASE"]
            )

        return self.current_action