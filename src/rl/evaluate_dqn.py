import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.rl.dqn_agent import DQNAgent
from src.rl.sumo_single_agent_env import SingleIntersectionSumoEnv


MODEL_PATH = Path("results/rl/dqn_j1_single_agent.pt")
OUTPUT_DIR = Path("results/rl")

DETAILS_PATH = OUTPUT_DIR / "dqn_j1_evaluation.csv"
SUMMARY_PATH = OUTPUT_DIR / "dqn_j1_evaluation_summary.csv"

SCENARIOS = ("low", "medium", "high")
EVALUATION_SEEDS = (301, 302, 303)

ACTION_TO_INDEX = {
    "NS_GREEN": 0,
    "EW_GREEN": 1,
}


def evaluate_episode(policy, scenario, seed, agent=None):
    env = SingleIntersectionSumoEnv(
        scenario=scenario,
        seed=seed,
        controlled_intersection="J1",
    )

    try:
        observation, reset_info = env.reset(seed=seed)

        total_reward = 0.0
        queue_sum = 0.0
        waiting_sum = 0.0
        switch_count = 0
        steps = 0
        done = False

        while not done:
            if policy == "DQN":
                action = agent.select_action(
                    observation,
                    explore=False,
                )
            elif policy == "Pressure":
                belief = SimpleNamespace(state=env.last_state)
                desired_action = env.controllers["J1"].choose_action(
                    belief
                )
                action = ACTION_TO_INDEX[desired_action]
            else:
                raise ValueError(f"Unknown policy: {policy}")

            observation, reward, done, info = env.step(action)

            total_reward += reward
            queue_sum += info["total_queue"]
            waiting_sum += info["total_waiting"]
            switch_count += int(info["switch_started"])
            steps += 1

        if steps == 0:
            raise RuntimeError(
                f"No evaluation steps completed: {policy}, "
                f"{scenario}, seed {seed}"
            )

        return {
            "policy": policy,
            "scenario": scenario,
            "seed": seed,
            "steps": steps,
            "episode_reward": total_reward,
            "average_queue": queue_sum / steps,
            "average_waiting": waiting_sum / steps,
            "total_arrivals": env.episode_arrivals,
            "signal_switches": switch_count,
        }

    finally:
        env.close()


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    if DETAILS_PATH.exists() or SUMMARY_PATH.exists():
        raise FileExistsError(
            "Evaluation output already exists. "
            "Existing results will not be overwritten."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    agent = DQNAgent(
        state_dim=12,
        action_dim=2,
        seed=999,
    )
    agent.load(str(MODEL_PATH))
    agent.online_network.eval()

    print("=" * 68)
    print("PAIRED DQN VS PRESSURE-BASED EVALUATION")
    print("=" * 68)
    print(f"Scenarios: {list(SCENARIOS)}")
    print(f"Unseen evaluation seeds: {list(EVALUATION_SEEDS)}")
    print("Policies: DQN, Pressure")
    print("Controlled intersection: J1")
    print("Other intersections: Pressure-based")
    print()

    records = []

    for scenario in SCENARIOS:
        for seed in EVALUATION_SEEDS:
            for policy in ("DQN", "Pressure"):
                result = evaluate_episode(
                    policy=policy,
                    scenario=scenario,
                    seed=seed,
                    agent=agent,
                )
                records.append(result)

                print(
                    f"{policy:8s} | {scenario:6s} | seed {seed} | "
                    f"steps {result['steps']:3d} | "
                    f"queue {result['average_queue']:7.3f} | "
                    f"waiting {result['average_waiting']:8.3f} | "
                    f"arrivals {result['total_arrivals']:4d} | "
                    f"switches {result['signal_switches']:3d}"
                )

    details = pd.DataFrame(records)
    details.to_csv(DETAILS_PATH, index=False)

    summary = (
        details.groupby(["policy", "scenario"], as_index=False)
        .agg(
            episodes=("seed", "count"),
            average_queue=("average_queue", "mean"),
            average_waiting=("average_waiting", "mean"),
            total_arrivals=("total_arrivals", "mean"),
            episode_reward=("episode_reward", "mean"),
            signal_switches=("signal_switches", "mean"),
        )
    )
    summary.to_csv(SUMMARY_PATH, index=False)

    overall = (
        details.groupby("policy", as_index=False)
        .agg(
            episodes=("seed", "count"),
            average_queue=("average_queue", "mean"),
            average_waiting=("average_waiting", "mean"),
            total_arrivals=("total_arrivals", "mean"),
            episode_reward=("episode_reward", "mean"),
            signal_switches=("signal_switches", "mean"),
        )
    )

    config = {
        "model": str(MODEL_PATH),
        "controlled_intersection": "J1",
        "scenarios": list(SCENARIOS),
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "policies": ["DQN", "Pressure"],
        "training_seeds": list(range(101, 131)),
        "training_updates_during_evaluation": False,
        "metrics": [
            "average_queue",
            "average_waiting",
            "total_arrivals",
            "episode_reward",
            "signal_switches",
        ],
    }

    config_path = OUTPUT_DIR / "dqn_j1_evaluation_config.json"
    if config_path.exists():
        raise FileExistsError(
            f"Evaluation config already exists: {config_path}"
        )

    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=4)

    print()
    print("=" * 68)
    print("SCENARIO-WISE EVALUATION SUMMARY")
    print("=" * 68)
    print(summary.to_string(index=False))

    print()
    print("=" * 68)
    print("OVERALL POLICY COMPARISON")
    print("=" * 68)
    print(overall.to_string(index=False))

    print()
    print("Evaluation completed.")
    print(f"Detailed results: {DETAILS_PATH}")
    print(f"Summary results:  {SUMMARY_PATH}")
    print(f"Config:           {config_path}")


if __name__ == "__main__":
    main()
