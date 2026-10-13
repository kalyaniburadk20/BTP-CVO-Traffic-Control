import csv
import os


class NetworkAcquisitionLogger:
    """
    Save observation-acquisition decisions for
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
            "policy",
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
        policy,
        decision
    ):
        """
        Save one acquisition decision.

        decision:
            True  -> ACQUIRE
            False -> SKIP
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
                "policy": policy,
                "decision": decision_text
            })