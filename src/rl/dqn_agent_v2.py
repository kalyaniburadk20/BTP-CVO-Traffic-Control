from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class SMDPReplayBuffer:
    """Replay buffer storing transitions with duration-aware discounts."""

    def __init__(self, capacity=50000, seed=42):
        if capacity <= 0:
            raise ValueError("Replay capacity must be positive.")

        self.memory = deque(maxlen=capacity)
        self.random = random.Random(seed)

    def add(self, state, action, reward, next_state, done, discount):
        self.memory.append((
            np.asarray(state, dtype=np.float32).copy(),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32).copy(),
            bool(done),
            float(discount),
        ))

    def sample(self, batch_size):
        if batch_size > len(self.memory):
            raise ValueError("Not enough replay samples.")

        batch = self.random.sample(list(self.memory), batch_size)
        states, actions, rewards, next_states, dones, discounts = zip(*batch)

        return (
            np.stack(states),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_states),
            np.asarray(dones, dtype=np.float32),
            np.asarray(discounts, dtype=np.float32),
        )

    def __len__(self):
        return len(self.memory)


class DuelingQNetwork(nn.Module):
    """Dueling network separating state value and action advantage."""

    def __init__(self, state_dim=12, action_dim=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, state):
        features = self.features(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        return value + advantage - advantage.mean(
            dim=1, keepdim=True
        )


class DQNAgentV2:
    """Double DQN + Dueling Network with duration-aware replay."""

    def __init__(
        self,
        state_dim=12,
        action_dim=2,
        learning_rate=0.0001,
        gamma=0.99,
        replay_capacity=50000,
        batch_size=64,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=5000,
        target_update_interval=100,
        seed=42,
        device=None,
    ):
        if state_dim <= 0 or action_dim <= 0:
            raise ValueError("State/action dimensions must be positive.")

        if not 0 < gamma <= 1:
            raise ValueError("Gamma must be in (0, 1].")

        if not 0 <= epsilon_end <= epsilon_start <= 1:
            raise ValueError("Invalid epsilon range.")

        if epsilon_decay_steps <= 0 or target_update_interval <= 0:
            raise ValueError("Update intervals must be positive.")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_update_interval = target_update_interval

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.random = random.Random(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self.online_network = DuelingQNetwork(
            state_dim, action_dim
        ).to(self.device)

        self.target_network = DuelingQNetwork(
            state_dim, action_dim
        ).to(self.device)

        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )
        self.target_network.eval()

        self.optimizer = optim.Adam(
            self.online_network.parameters(),
            lr=learning_rate,
        )

        self.loss_function = nn.SmoothL1Loss()

        self.replay_buffer = SMDPReplayBuffer(
            replay_capacity, seed
        )

        self.environment_steps = 0
        self.gradient_steps = 0

    @property
    def epsilon(self):
        progress = min(
            self.environment_steps / self.epsilon_decay_steps,
            1.0,
        )

        return (
            self.epsilon_start
            + progress * (self.epsilon_end - self.epsilon_start)
        )

    def select_action(self, state, explore=True):
        state = np.asarray(state, dtype=np.float32)

        if state.shape != (self.state_dim,):
            raise ValueError(
                f"Expected state shape {(self.state_dim,)}, "
                f"received {state.shape}."
            )

        if not np.isfinite(state).all():
            raise ValueError("State contains non-finite values.")

        if explore and self.random.random() < self.epsilon:
            action = self.random.randrange(self.action_dim)
        else:
            state_tensor = torch.as_tensor(
                state,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            with torch.no_grad():
                q_values = self.online_network(state_tensor)

            action = int(q_values.argmax(dim=1).item())

        if explore:
            self.environment_steps += 1

        return action

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        discount,
    ):
        if action < 0 or action >= self.action_dim:
            raise ValueError("Action index is outside action space.")

        if not 0 <= discount <= 1:
            raise ValueError("Discount factor must be between 0 and 1.")

        self.replay_buffer.add(
            state,
            action,
            reward,
            next_state,
            done,
            discount,
        )

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            discounts,
        ) = self.replay_buffer.sample(self.batch_size)

        states = torch.as_tensor(
            states, dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            actions, dtype=torch.long, device=self.device
        )
        rewards = torch.as_tensor(
            rewards, dtype=torch.float32, device=self.device
        )
        next_states = torch.as_tensor(
            next_states, dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(
            dones, dtype=torch.float32, device=self.device
        )
        discounts = torch.as_tensor(
            discounts, dtype=torch.float32, device=self.device
        )

        current_q_values = self.online_network(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        # Double DQN:
        # Online network selects the next action.
        # Target network evaluates that selected action.
        with torch.no_grad():
            next_actions = self.online_network(
                next_states
            ).argmax(dim=1, keepdim=True)

            next_q_values = self.target_network(
                next_states
            ).gather(1, next_actions).squeeze(1)

            target_q_values = (
                rewards
                + discounts * (1.0 - dones) * next_q_values
            )

        loss = self.loss_function(
            current_q_values, target_q_values
        )

        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(
            self.online_network.parameters(),
            max_norm=5.0,
        )

        self.optimizer.step()
        self.gradient_steps += 1

        if self.gradient_steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(
                self.online_network.state_dict()
            )

        return float(loss.item())

    def save(self, path):
        checkpoint = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "online_network": self.online_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "environment_steps": self.environment_steps,
            "gradient_steps": self.gradient_steps,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "gamma": self.gamma,
        }

        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )

        if checkpoint["state_dim"] != self.state_dim:
            raise ValueError("Checkpoint state dimension mismatch.")

        if checkpoint["action_dim"] != self.action_dim:
            raise ValueError("Checkpoint action dimension mismatch.")

        self.online_network.load_state_dict(
            checkpoint["online_network"]
        )
        self.target_network.load_state_dict(
            checkpoint["target_network"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        self.environment_steps = checkpoint["environment_steps"]
        self.gradient_steps = checkpoint["gradient_steps"]
