import csv
import os


class CVOLogger:
    """
    Saves CVO calculation results to a CSV file.
    """

    def __init__(self, output_file):
        self.output_file = output_file

        parent_dir = os.path.dirname(output_file)

        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        with open(
            self.output_file,
            "w",
            newline="",
            encoding="utf-8"
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "time",
                    "current_belief",
                    "hypothetical_belief",
                    "current_action",
                    "hypothetical_action",
                    "decision_margin",
                    "cvo"
                ]
            )

            writer.writeheader()

    def log(
        self,
        time,
        current_belief,
        hypothetical_belief,
        current_action,
        hypothetical_action,
        decision_margin,
        cvo
    ):
        """
        Append one CVO calculation to the CSV log.
        """

        with open(
            self.output_file,
            "a",
            newline="",
            encoding="utf-8"
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "time",
                    "current_belief",
                    "hypothetical_belief",
                    "current_action",
                    "hypothetical_action",
                    "decision_margin",
                    "cvo"
                ]
            )

            writer.writerow({
                "time": time,
                "current_belief": current_belief,
                "hypothetical_belief": hypothetical_belief,
                "current_action": current_action,
                "hypothetical_action": hypothetical_action,
                "decision_margin": decision_margin,
                "cvo": cvo
            })