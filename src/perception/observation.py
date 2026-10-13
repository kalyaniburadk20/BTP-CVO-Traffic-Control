# --------------------------------------------------
# OBSERVATION REPRESENTATION
# --------------------------------------------------

class Observation:
    """
    Represents one acquired observation of traffic
    information at a particular simulation time.
    """

    def __init__(self, traffic_state, timestamp):
        self.traffic_state = traffic_state
        self.timestamp = timestamp

    def to_dict(self):
        """
        Convert observation into a simple dictionary.
        Useful for debugging and logging.
        """
        return {
            "traffic_state": self.traffic_state.to_dict(),
            "timestamp": self.timestamp,
        }