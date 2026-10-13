import os
import sys


# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.acquisition.config_loader import (
    load_acquisition_config
)


def main():

    print()
    print("=" * 75)
    print("PHASE 5 — CONFIGURATION LOADER TEST")
    print("=" * 75)

    config = load_acquisition_config()

    print()
    print("Loaded configuration:")
    print()

    print(
        f"Periodic interval: "
        f"{config['periodic']['interval']}"
    )

    print(
        f"State-Change threshold: "
        f"{config['state_change']['threshold']}"
    )

    print(
        f"AoI max age: "
        f"{config['aoi']['max_age']}"
    )

    print()
    print("-" * 75)

    # Basic validation
    assert config["periodic"]["interval"] > 0
    assert config["state_change"]["threshold"] >= 0
    assert config["aoi"]["max_age"] > 0

    print(
        "PASS — Configuration loaded successfully."
    )

    print("=" * 75)


if __name__ == "__main__":
    main()