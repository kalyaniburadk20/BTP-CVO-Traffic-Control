import csv
import os


class TrafficLogger:
    """
    Save per-step traffic data for one experiment run.
    """

    def __init__(self, output_file):
        self.output_file = output_file

        parent_dir = os.path.dirname(output_file)

        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        self.fieldnames = [
            "time",
            "north_queue",
            "south_queue",
            "east_queue",
            "west_queue",
            "north_waiting",
            "south_waiting",
            "east_waiting",
            "west_waiting",
            "signal_phase",
            "vehicle_count",
            "arrived_this_step"
        ]

        with open(
            self.output_file,
            "w",
            newline="",
            encoding="utf-8"
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=self.fieldnames
            )

            writer.writeheader()

    def log(
        self,
        time,
        state,
        arrived_this_step
    ):
        """
        Save one simulation-step traffic record.
        """

        with open(
            self.output_file,
            "a",
            newline="",
            encoding="utf-8"
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=self.fieldnames
            )

            writer.writerow({
                "time": time,
                "north_queue": state.north_queue,
                "south_queue": state.south_queue,
                "east_queue": state.east_queue,
                "west_queue": state.west_queue,
                "north_waiting": state.north_waiting,
                "south_waiting": state.south_waiting,
                "east_waiting": state.east_waiting,
                "west_waiting": state.west_waiting,
                "signal_phase": state.current_phase,
                "vehicle_count": state.vehicle_count,
                "arrived_this_step": arrived_this_step
            })