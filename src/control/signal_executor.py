import traci


class SignalExecutor:
    """
    Executes the controller's desired signal action.

    Enforces:
    1. Minimum green time
    2. Yellow transition before switching

    SUMO phases:
        0 -> NS_GREEN
        1 -> NS_YELLOW
        2 -> EW_GREEN
        3 -> EW_YELLOW
    """

    MIN_GREEN_TIME = 10.0

    NS_GREEN_PHASE = 0
    NS_YELLOW_PHASE = 1

    EW_GREEN_PHASE = 2
    EW_YELLOW_PHASE = 3

    def __init__(self):

        self.current_action = None

        self.phase_start_time = None

        # Action that will become active after yellow
        self.pending_action = None

        # Whether we are currently inside a yellow transition
        self.in_yellow = False

    # --------------------------------------------------
    # INITIALIZATION
    # --------------------------------------------------

    def initialize(self):
        """
        Initialize executor from the current SUMO phase.
        """

        current_phase = traci.trafficlight.getPhase("J0")
        current_time = traci.simulation.getTime()

        self.current_action = self.phase_to_action(
            current_phase
        )

        self.phase_start_time = current_time

        self.pending_action = None
        self.in_yellow = False

    # --------------------------------------------------
    # PHASE -> ACTION
    # --------------------------------------------------

    def phase_to_action(self, phase):
        """
        Convert SUMO phase into our action representation.
        """

        if phase in (
            self.NS_GREEN_PHASE,
            self.NS_YELLOW_PHASE
        ):
            return "NS_GREEN"

        return "EW_GREEN"

    # --------------------------------------------------
    # ELAPSED GREEN TIME
    # --------------------------------------------------

    def get_elapsed_green_time(self, current_time):
        """
        Return elapsed time since current green action
        became active.
        """

        if self.phase_start_time is None:
            return 0.0

        return current_time - self.phase_start_time

    # --------------------------------------------------
    # APPLY ACTION
    # --------------------------------------------------

    def apply_action(self, desired_action):
        """
        Apply the desired action while respecting:
        - minimum green time
        - yellow transition
        """

        current_time = traci.simulation.getTime()

        if self.current_action is None:
            self.initialize()

        # --------------------------------------------------
        # CASE 1: Currently in yellow transition
        # --------------------------------------------------

        if self.in_yellow:

            current_phase = traci.trafficlight.getPhase("J0")

            # Check whether yellow phase has finished
            if current_phase not in (
                self.NS_YELLOW_PHASE,
                self.EW_YELLOW_PHASE
            ):

                self.in_yellow = False

                # Pending action has become active
                self.current_action = self.phase_to_action(
                    current_phase
                )

                self.phase_start_time = current_time

                self.pending_action = None

                return self.current_action

            # Yellow is still running
            return self.current_action

        # --------------------------------------------------
        # CASE 2: Desired action is already active
        # --------------------------------------------------

        if desired_action == self.current_action:
            return self.current_action

        # --------------------------------------------------
        # CASE 3: Minimum green time not completed
        # --------------------------------------------------

        elapsed_time = self.get_elapsed_green_time(
            current_time
        )

        if elapsed_time < self.MIN_GREEN_TIME:
            return self.current_action

        # --------------------------------------------------
        # CASE 4: Start yellow transition
        # --------------------------------------------------

        self.pending_action = desired_action

        if self.current_action == "NS_GREEN":

            traci.trafficlight.setPhase(
                "J0",
                self.NS_YELLOW_PHASE
            )

        elif self.current_action == "EW_GREEN":

            traci.trafficlight.setPhase(
                "J0",
                self.EW_YELLOW_PHASE
            )

        else:
            raise ValueError(
                f"Unknown current action: "
                f"{self.current_action}"
            )

        self.in_yellow = True

        return self.current_action