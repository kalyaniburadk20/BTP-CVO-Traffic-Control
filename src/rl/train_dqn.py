import json
from pathlib import Path

import pandas as pd

from src.rl.dqn_agent import DQNAgent
from src.rl.sumo_single_agent_env import SingleIntersectionSumoEnv


# --------------------------------------------------
# TRAINING CONFIGURATION
# --------------------------------------------------

NUM_EPISODES = 30

SCENARIOS = [
    "low",
    "medium",
    "high",
]

CONTROLLED_INTERSECTION = "J1"

TRAINING_SEED_START = 101

AGENT_SEED = 42

BATCH_SIZE = 64

OUTPUT_DIR = Path("results/rl")

MODEL_PATH = OUTPUT_DIR / "dqn_j1_single_agent.pt"
METRICS_PATH = OUTPUT_DIR / "dqn_j1_training_metrics.csv"
CONFIG_PATH = OUTPUT_DIR / "dqn_j1_training_config.json"


def main():
    for path in [MODEL_PATH, METRICS_PATH, CONFIG_PATH]:
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing output: {path}"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = [
        SCENARIOS[episode % len(SCENARIOS)]
        for episode in range(NUM_EPISODES)
    ]

    agent = DQNAgent(
        state_dim=12,
        action_dim=2,
        learning_rate=0.001,
        gamma=0.99,
        replay_capacity=50000,
        batch_size=BATCH_SIZE,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=16000,
        target_update_interval=500,
        seed=AGENT_SEED,
    )

    results = []

    print("=" * 70)
    print("SINGLE-AGENT DQN TRAINING")
    print("=" * 70)
    print(f"Controlled intersection: {CONTROLLED_INTERSECTION}")
    print(f"Episodes: {NUM_EPISODES}")
    print(f"Scenarios: {SCENARIOS}")
    print(f"Training device: {agent.device}")
    print(f"Batch size: {BATCH_SIZE}")
    print()

    for episode_index, scenario in enumerate(scenarios):
        episode_number = episode_index + 1
        episode_seed = TRAINING_SEED_START + episode_index

        env = SingleIntersectionSumoEnv(
            scenario=scenario,
            seed=episode_seed,
            controlled_intersection=CONTROLLED_INTERSECTION,
        )

        losses = []
        queue_sum = 0.0
        waiting_sum = 0.0
        arrivals = 0
        switches = 0
        episode_reward = 0.0
        steps = 0

        try:
            state, reset_info = env.reset()

            done = False

            max_steps = int(
                env.duration - env.warmup
            ) + 5

            while not done:
                action = agent.select_action(
                    state,
                    explore=True,
                )

                next_state, reward, done, info = env.step(
                    action
                )

                agent.remember(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                )

                loss = agent.train_step()

                if loss is not None:
                    losses.append(loss)

                state = next_state

                steps += 1
                episode_reward += reward
                queue_sum += info["total_queue"]
                waiting_sum += info["total_waiting"]
                arrivals += info["arrived_this_step"]

                if info["switch_started"]:
                    switches += 1

                if steps > max_steps:
                    raise RuntimeError(
                        f"Episode {episode_number} exceeded "
                        f"maximum expected steps."
                    )

        finally:
            env.close()

        average_queue = (
            queue_sum / steps if steps else 0.0
        )
        average_waiting = (
            waiting_sum / steps if steps else 0.0
        )
        average_loss = (
            sum(losses) / len(losses) if losses else None
        )

        result = {
            "episode": episode_number,
            "scenario": scenario,
            "seed": episode_seed,
            "intersection": CONTROLLED_INTERSECTION,
            "steps": steps,
            "episode_reward": episode_reward,
            "average_queue": average_queue,
            "average_waiting": average_waiting,
            "total_arrivals": arrivals,
            "switches": switches,
            "mean_training_loss": average_loss,
            "epsilon": agent.epsilon,
            "gradient_steps": agent.gradient_steps,
        }

        results.append(result)

        print(
            f"Episode {episode_number:02d}/{NUM_EPISODES} | "
            f"Scenario: {scenario:6s} | "
            f"Steps: {steps:3d} | "
            f"Reward: {episode_reward:9.3f} | "
            f"Avg Queue: {average_queue:7.3f} | "
            f"Avg Waiting: {average_waiting:8.3f} | "
            f"Arrivals: {arrivals:4d} | "
            f"Switches: {switches:3d} | "
            f"Epsilon: {agent.epsilon:.3f}"
        )

    metrics = pd.DataFrame(results)
    metrics.to_csv(METRICS_PATH, index=False)

    agent.save(MODEL_PATH)

    training_config = {
        "algorithm": "Deep Q-Network",
        "controlled_intersection": CONTROLLED_INTERSECTION,
        "episodes": NUM_EPISODES,
        "scenarios": SCENARIOS,
        "training_seed_start": TRAINING_SEED_START,
        "agent_seed": AGENT_SEED,
        "state_dimension": 12,
        "action_dimension": 2,
        "batch_size": BATCH_SIZE,
        "learning_rate": 0.001,
        "gamma": 0.99,
        "replay_capacity": 50000,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": 16000,
        "target_update_interval": 500,
        "training_device": str(agent.device),
        "warmup_seconds": 60,
        "simulation_duration_seconds": 600,
        "background_control": "Queue-pressure-based controller",
    }

    with CONFIG_PATH.open("w", encoding="utf-8") as file:
        json.dump(training_config, file, indent=4)

    print("\n" + "=" * 70)
    print("DQN TRAINING COMPLETE")
    print("=" * 70)
    print("Total episodes:", len(metrics))
    print("Total environment steps:", int(metrics["steps"].sum()))
    print("Total gradient steps:", agent.gradient_steps)
    print("Final exploration epsilon:", round(agent.epsilon, 4))

    print("\nScenario-wise training summary:")
    print(
        metrics.groupby("scenario")[
            ["episode_reward", "average_queue",
             "average_waiting", "total_arrivals"]
        ].mean().round(4).to_string()
    )

    print("\nSaved files:")
    print(MODEL_PATH)
    print(METRICS_PATH)
    print(CONFIG_PATH)


if __name__ == "__main__":
    main()
