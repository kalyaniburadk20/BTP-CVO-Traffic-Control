import os
import yaml


# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


# --------------------------------------------------
# CONFIG FILE
# --------------------------------------------------

CONFIG_FILE = os.path.join(
    PROJECT_ROOT,
    "configs",
    "acquisition.yaml"
)


def load_acquisition_config():
    """
    Load acquisition-policy parameters from YAML.
    """

    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(
            f"Configuration file not found: "
            f"{CONFIG_FILE}"
        )

    with open(
        CONFIG_FILE,
        "r",
        encoding="utf-8"
    ) as config_file:

        config = yaml.safe_load(
            config_file
        )

    return config