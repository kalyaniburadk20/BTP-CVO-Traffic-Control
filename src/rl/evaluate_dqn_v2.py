import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.rl.dqn_agent_v2 import DQNAgentV2
from src.rl.sumo_single_agent_env import SingleIntersectionSumoEnv


MODEL_PATH = Path("results/rl/dqn_j1_single_agent_v2.pt")
OUTPUT_DIR = Path("results/rl")

DETAILS_PATH = OUTPUT_DIR / "dqn_j1_evaluation_v2.csv"
SUMMARY_PATH = OUTPUT_DIR / "dqn_j1_evaluation_summary_v2.csv"
CONFIG_PATH = OUTPUT_DIR / "dqn_j1_evaluation_config_v2.json"

SCENARIOS = ("low", "medium", "high")
EVALUATION_SEEDS = (501, 502, 503)
DECISION_INTERVAL = 10

POLICIES = (
    "DQN_V2",
    "Pressure_Matched_10s",
    "Pressure_Native_1s",
)

ACTION_TO_INDEX = {
    "NS_GREEN": 0,
    "EW_GREEN": 1,
}


def get_pressure_action(env):
    belief = SimpleNamespace(state=env.last_state)

    desired_action = env.controllers["J1"].choose_action(
        belief
    )

    return ACTION_TO_INDEX[desired_action]


def evaluate_episode(policy, scenario, seed, agent):
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
        decision_steps = 0
        done = False

        while not done:

            if policy == "DQN_V2":
                action = agent.select_action(
                    observation,
                    explore=False,
                )
                block_length = DECISION_INTERVAL

            elif policy == "Pressure_Matched_10s":
                action = get_pressure_action(env)
                block_length = DECISION_INTERVAL

            elif policy == "Pressure_Native_1s":
                action = get_pressure_action(env)
                block_length = 1

            else:
                raise ValueError(
                    f"Unknown policy: {policy}"
                )

            decision_steps += 1

            for _ in range(block_length):
                (
                    observation,
                    reward,
                    done,
                    info,
                ) = env.step(action)

                total_reward += reward
                queue_sum += info["total_queue"]
                waiting_sum += info["total_waiting"]
                switch_count += int(info["switch_started"])
                steps += 1

                if done:
                    break

        if steps == 0:
            raise RuntimeError(
                f"No steps completed: {policy}, "
                f"{scenario}, seed {seed}"
            )

        return {
            "policy": policy,
            "scenario": scenario,
            "seed": seed,
            "steps": steps,
            "decision_steps": decision_steps,
            "episode_reward": total_reward,
            "average_queue": queue_sum / steps,
            "average_waiting": waiting_sum / steps,
            "total_arrivals": env.episode_arrivals,
            "signal_switches": switch_count,
        }

    finally:
        env.close()


def main():
    for path in (MODEL_PATH,):
        if not path.exists():
            raise FileNotFoundError(
                f"Required model not found: {path}"
            )

    for path in (DETAILS_PATH, SUMMARY_PATH, CONFIG_PATH):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing output: {path}"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    agent = DQNAgentV2(
        state_dim=12,
        action_dim=2,
        seed=999,
    )

    agent.load(str(MODEL_PATH))
    agent.online_network.eval()

    print("=" * 72)
    print("PAIRED DQN V2 EVALUATION")
    print("=" * 72)
    print("Scenarios:", list(SCENARIOS))
    print("Unseen evaluation seeds:", list(EVALUATION_SEEDS))
    print("Controlled intersection: J1")
    print("Decision interval for DQN V2: 10 seconds")
    print("Policies:", list(POLICIES))
    print()

    records = []

    for scenario in SCENARIOS:
        for seed in EVALUATION_SEEDS:
            for policy in POLICIES:
                result = evaluate_episode(
                    policy=policy,
                    scenario=scenario,
                    seed=seed,
                    agent=agent,
                )

                records.append(result)

                print(
                    f"{policy:22s} | "
                    f"{scenario:6s} | "
                    f"seed {seed} | "
                    f"steps {result['steps']:3d} | "
                    f"decisions {result['decision_steps']:3d} | "
                    f"queue {result['average_queue']:7.3f} | "
                    f"waiting {result['average_waiting']:8.3f} | "
                    f"arrivals {result['total_arrivals']:4d} | "
                    f"switches {result['signal_switches']:3d}"
                )

    details = pd.DataFrame(records)
    details.to_csv(DETAILS_PATH, index=False)

    summary = (
        details.groupby(
            ["policy", "scenario"],
            as_index=False,
        )
        .agg(
            episodes=("seed", "count"),
            average_queue=("average_queue", "mean"),
            average_waiting=("average_waiting", "mean"),
            total_arrivals=("total_arrivals", "mean"),
            episode_reward=("episode_reward", "mean"),
            signal_switches=("signal_switches", "mean"),
            decision_steps=("decision_steps", "mean"),
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
        "algorithm": "Double DQN with Dueling Q-Network",
        "controlled_intersection": "J1",
        "scenarios": list(SCENARIOS),
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "training_seeds": list(range(401, 491)),
        "policies": list(POLICIES),
        "dqn_decision_interval_seconds": DECISION_INTERVAL,
        "pressure_matched_decision_interval_seconds": DECISION_INTERVAL,
        "pressure_native_decision_interval_seconds": 1,
        "training_updates_during_evaluation": False,
        "metrics": [
            "average_queue",
            "average_waiting",
            "total_arrivals",
            "episode_reward",
            "signal_switches",
        ],
    }

    with CONFIG_PATH.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=4)

    print()
    print("=" * 72)
    print("SCENARIO-WISE EVALUATION SUMMARY")
    print("=" * 72)
    print(summary.to_string(index=False))

    print()
    print("=" * 72)
    print("OVERALL POLICY COMPARISON")
    print("=" * 72)
    print(overall.to_string(index=False))

    print()
    print("DQN V2 evaluation completed.")
    print("Detailed results:", DETAILS_PATH)
    print("Summary results:", SUMMARY_PATH)
    print("Configuration:", CONFIG_PATH)


if __name__ == "__main__":
    main()
