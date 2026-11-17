import json
import math
from pathlib import Path

import pandas as pd

from src.rl.dqn_agent_v2 import DQNAgentV2
from src.rl.sumo_single_agent_env import SingleIntersectionSumoEnv


NUM_EPISODES = 90
SCENARIOS = ("low", "medium", "high")
CONTROLLED_INTERSECTION = "J1"

TRAINING_SEED_START = 401
AGENT_SEED = 43

DECISION_INTERVAL = 10
GAMMA = 0.99
EPSILON_DECAY_STEPS = 4000

OUTPUT_DIR = Path("results/rl")
MODEL_PATH = OUTPUT_DIR / "dqn_j1_single_agent_v2.pt"
METRICS_PATH = OUTPUT_DIR / "dqn_j1_training_metrics_v2.csv"
CONFIG_PATH = OUTPUT_DIR / "dqn_j1_training_config_v2.json"


def main():
    for path in (MODEL_PATH, METRICS_PATH, CONFIG_PATH):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing output: {path}"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    agent = DQNAgentV2(
        state_dim=12,
        action_dim=2,
        learning_rate=0.0001,
        gamma=GAMMA,
        replay_capacity=50000,
        batch_size=64,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=EPSILON_DECAY_STEPS,
        target_update_interval=200,
        seed=AGENT_SEED,
    )

    results = []
    total_environment_steps = 0

    print("=" * 70)
    print("SINGLE-AGENT DOUBLE DUELING DQN V2 TRAINING")
    print("=" * 70)
    print("Controlled intersection:", CONTROLLED_INTERSECTION)
    print("Episodes:", NUM_EPISODES)
    print("Scenarios:", list(SCENARIOS))
    print("Decision interval:", DECISION_INTERVAL, "seconds")
    print("Training device:", agent.device)
    print("Batch size:", agent.batch_size)
    print("Epsilon decay decisions:", EPSILON_DECAY_STEPS)
    print()

    for episode_index in range(NUM_EPISODES):
        episode_number = episode_index + 1
        scenario = SCENARIOS[episode_index % len(SCENARIOS)]
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
        environment_steps = 0
        decision_steps = 0

        try:
            state, reset_info = env.reset(seed=episode_seed)
            done = False

            max_environment_steps = int(
                env.duration - env.warmup
            ) + 5

            while not done:
                action = agent.select_action(
                    state,
                    explore=True,
                )

                discounted_return = 0.0
                duration = 0
                next_state = state

                for _ in range(DECISION_INTERVAL):
                    next_state, reward, done, info = env.step(action)

                    discounted_return += (
                        GAMMA ** duration
                    ) * reward

                    duration += 1
                    environment_steps += 1

                    episode_reward += reward
                    queue_sum += info["total_queue"]
                    waiting_sum += info["total_waiting"]
                    arrivals += info["arrived_this_step"]

                    if info["switch_started"]:
                        switches += 1

                    if done:
                        break

                # Duration-aware SMDP transition.
                transition_discount = GAMMA ** duration

                agent.remember(
                    state,
                    action,
                    discounted_return,
                    next_state,
                    done,
                    transition_discount,
                )

                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)

                state = next_state
                decision_steps += 1

                if environment_steps > max_environment_steps:
                    raise RuntimeError(
                        f"Episode {episode_number} exceeded "
                        "maximum expected environment steps."
                    )

        finally:
            env.close()

        total_environment_steps += environment_steps

        average_queue = (
            queue_sum / environment_steps
            if environment_steps else 0.0
        )
        average_waiting = (
            waiting_sum / environment_steps
            if environment_steps else 0.0
        )
        average_loss = (
            sum(losses) / len(losses)
            if losses else None
        )

        result = {
            "episode": episode_number,
            "scenario": scenario,
            "seed": episode_seed,
            "intersection": CONTROLLED_INTERSECTION,
            "decision_interval": DECISION_INTERVAL,
            "decision_steps": decision_steps,
            "environment_steps": environment_steps,
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
            f"Decisions: {decision_steps:3d} | "
            f"Reward: {episode_reward:9.3f} | "
            f"Queue: {average_queue:7.3f} | "
            f"Waiting: {average_waiting:8.3f} | "
            f"Arrivals: {arrivals:4d} | "
            f"Switches: {switches:3d} | "
            f"Epsilon: {agent.epsilon:.3f}"
        )

    metrics = pd.DataFrame(results)
    metrics.to_csv(METRICS_PATH, index=False)
    agent.save(str(MODEL_PATH))

    config = {
        "algorithm": "Double DQN with Dueling Q-Network",
        "replay_type": "Duration-aware SMDP replay",
        "controlled_intersection": CONTROLLED_INTERSECTION,
        "episodes": NUM_EPISODES,
        "scenarios": list(SCENARIOS),
        "training_seed_start": TRAINING_SEED_START,
        "agent_seed": AGENT_SEED,
        "state_dimension": 12,
        "action_dimension": 2,
        "decision_interval_seconds": DECISION_INTERVAL,
        "learning_rate": 0.0001,
        "gamma": GAMMA,
        "replay_capacity": 50000,
        "batch_size": 64,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_decisions": EPSILON_DECAY_STEPS,
        "target_update_interval": 200,
        "training_device": str(agent.device),
        "warmup_seconds": 60,
        "simulation_duration_seconds": 600,
        "background_control": "Queue-pressure-based controller",
        "reward_weights": {
            "queue": 0.10,
            "waiting": 0.001,
            "switching_penalty": 0.5,
        },
    }

    with CONFIG_PATH.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=4)

    print()
    print("=" * 70)
    print("DQN V2 TRAINING COMPLETE")
    print("=" * 70)
    print("Episodes:", len(metrics))
    print("Environment steps:", total_environment_steps)
    print("Decision steps:", int(metrics["decision_steps"].sum()))
    print("Gradient steps:", agent.gradient_steps)
    print("Final epsilon:", round(agent.epsilon, 4))

    print("\nScenario-wise training summary:")
    print(
        metrics.groupby("scenario")[
            [
                "episode_reward",
                "average_queue",
                "average_waiting",
                "total_arrivals",
                "switches",
            ]
        ].mean().round(4).to_string()
    )

    print("\nSaved files:")
    print(MODEL_PATH)
    print(METRICS_PATH)
    print(CONFIG_PATH)


if __name__ == "__main__":
    main()
