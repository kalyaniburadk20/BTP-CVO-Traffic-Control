from src.acquisition.base_policy import AcquisitionPolicy
from src.acquisition.config_loader import load_acquisition_config


class PeriodicPolicy(AcquisitionPolicy):
    """
    Time-based acquisition policy.

    Acquire once every fixed interval.
    """

    def __init__(self, interval=None):

        # If no interval is provided,
        # load it from acquisition.yaml.
        if interval is None:
            config = load_acquisition_config()

            interval = config["periodic"]["interval"]

        if interval <= 0:
            raise ValueError(
                "Interval must be greater than 0."
            )

        self.interval = interval

    def should_acquire(
        self,
        current_time,
        belief,
        observation_history
    ):
        """
        Acquire when current time is an exact
        multiple of the configured interval.
        """

        return current_time % self.interval == 0