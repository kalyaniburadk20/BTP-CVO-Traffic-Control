import csv
import os


class AcquisitionLogger:
    """
    Common logger for observation-acquisition decisions.
    """

    def __init__(self, output_file):
        self.output_file = output_file

        # Create parent directory if needed
        parent_dir = os.path.dirname(output_file)

        if parent_dir:
            os.makedirs(
                parent_dir,
                exist_ok=True
            )

        # Create file and write header
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
                    "policy",
                    "decision"
                ]
            )

            writer.writeheader()

    def log(
        self,
        time,
        policy,
        decision
    ):
        """
        Write one acquisition decision.

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
                fieldnames=[
                    "time",
                    "policy",
                    "decision"
                ]
            )

            writer.writerow({
                "time": time,
                "policy": policy,
                "decision": decision_text
            })