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


from src.acquisition.periodic import PeriodicPolicy
from src.acquisition.state_change import StateChangePolicy
from src.acquisition.aoi import AoIPolicy


def main():

    print()
    print("=" * 80)
    print("PHASE 5 — CONFIGURED POLICIES TEST")
    print("=" * 80)

    # --------------------------------------------------
    # Create policies WITHOUT passing parameters.
    # They must read them from acquisition.yaml.
    # --------------------------------------------------

    periodic = PeriodicPolicy()
    state_change = StateChangePolicy()
    aoi = AoIPolicy()

    print()
    print("Loaded policy parameters:")
    print()

    print(
        f"Periodic interval       : "
        f"{periodic.interval}"
    )

    print(
        f"State-Change threshold  : "
        f"{state_change.threshold}"
    )

    print(
        f"AoI max age             : "
        f"{aoi.max_age}"
    )

    print()
    print("-" * 80)

    # --------------------------------------------------
    # Validate configured values
    # --------------------------------------------------

    assert periodic.interval == 5
    assert state_change.threshold == 10
    assert aoi.max_age == 10

    print(
        "PASS — All policies loaded configuration correctly."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()