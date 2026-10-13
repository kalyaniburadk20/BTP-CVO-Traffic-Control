import csv
import os
import json


class NetworkCVOLogger:
    """
    Save CVO calculation results for each
    intersection in the network.
    """

    def __init__(self, output_file):
        self.output_file = output_file

        parent_dir = os.path.dirname(output_file)

        if parent_dir:
            os.makedirs(
                parent_dir,
                exist_ok=True
            )

        self.fieldnames = [
            "time",
            "intersection_id",
            "current_belief",
            "hypothetical_belief",
            "current_action",
            "hypothetical_action",
            "decision_margin",
            "cvo"
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
        intersection_id,
        current_belief,
        hypothetical_belief,
        current_action,
        hypothetical_action,
        decision_margin,
        cvo
    ):
        """
        Save one CVO calculation for one
        intersection at one simulation timestep.
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

            current_state = current_belief.state
            hypothetical_state = hypothetical_belief.state

            current_belief_data = {
                "north_queue": current_state.north_queue,
                "south_queue": current_state.south_queue,
                "east_queue": current_state.east_queue,
                "west_queue": current_state.west_queue,
                "north_waiting": current_state.north_waiting,
                "south_waiting": current_state.south_waiting,
                "east_waiting": current_state.east_waiting,
                "west_waiting": current_state.west_waiting,
                "current_phase": current_state.current_phase,
                "vehicle_count": current_state.vehicle_count,
                "last_observation_time": (
                    current_belief.last_observation_time
                ),
            }

            hypothetical_belief_data = {
                "north_queue": hypothetical_state.north_queue,
                "south_queue": hypothetical_state.south_queue,
                "east_queue": hypothetical_state.east_queue,
                "west_queue": hypothetical_state.west_queue,
                "north_waiting": hypothetical_state.north_waiting,
                "south_waiting": hypothetical_state.south_waiting,
                "east_waiting": hypothetical_state.east_waiting,
                "west_waiting": hypothetical_state.west_waiting,
                "current_phase": hypothetical_state.current_phase,
                "vehicle_count": hypothetical_state.vehicle_count,
                "last_observation_time": (
                    hypothetical_belief.last_observation_time
                ),
            }

            writer.writerow({
                "time": time,
                "intersection_id": intersection_id,
                "current_belief": json.dumps(
    current_belief_data
),
"hypothetical_belief": json.dumps(
    hypothetical_belief_data
),
                "current_action": current_action,
                "hypothetical_action": hypothetical_action,
                "decision_margin": decision_margin,
                "cvo": cvo
            })