import csv
import os


class CVODecisionLogger:
    """
    Logger for complete CVO acquisition decisions.

    Each row stores:
    - current/hypothetical actions
    - CVO
    - raw cost components
    - normalized cost components
    - weighted total cost
    - final ACQUIRE/SKIP decision
    """

    def __init__(self, output_file):
        self.output_file = output_file

        parent_dir = os.path.dirname(output_file)

        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        self.fieldnames = [
            "time",
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
        current_action,
        hypothetical_action,
        cvo,
        cost_result,
        decision
    ):
        """
        Save one complete CVO acquisition decision.
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

                "current_action":
                    current_action,

                "hypothetical_action":
                    hypothetical_action,

                "cvo": cvo,

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