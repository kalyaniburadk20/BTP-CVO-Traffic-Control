import csv
import os


class NetworkDecisionLogger:
    """
    Save complete CVO acquisition decisions for
    each intersection in the network.
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
            "current_action",
            "hypothetical_action",
            "cvo",

            "raw_compute",
            "raw_communication",
            "raw_latency",

            "normalized_compute",
            "normalized_communication",
            "normalized_latency",

            "compute_weight",
            "communication_weight",
            "latency_weight",

            "total_cost",
            "decision"
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
        current_action,
        hypothetical_action,
        cvo,
        cost_result,
        decision
    ):
        """
        Save one complete CVO decision for one
        intersection at one simulation timestep.
        """

        decision_text = (
            "ACQUIRE"
            if decision
            else "SKIP"
        )

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
                "intersection_id": intersection_id,

                "current_action":
                    current_action,

                "hypothetical_action":
                    hypothetical_action,

                "cvo":
                    cvo,

                "raw_compute":
                    cost_result["raw_compute"],

                "raw_communication":
                    cost_result["raw_communication"],

                "raw_latency":
                    cost_result["raw_latency"],

                "normalized_compute":
                    cost_result["normalized_compute"],

                "normalized_communication":
                    cost_result["normalized_communication"],

                "normalized_latency":
                    cost_result["normalized_latency"],

                "compute_weight":
                    cost_result["compute_weight"],

                "communication_weight":
                    cost_result["communication_weight"],

                "latency_weight":
                    cost_result["latency_weight"],

                "total_cost":
                    cost_result["total_cost"],

                "decision":
                    decision_text
            })