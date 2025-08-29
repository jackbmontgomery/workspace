import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch import nn, optim


class DQN(nn.Module):
    def __init__(self, obs_dim, num_actions):
        super(DQN, self).__init__()

        self.layer_1 = nn.Linear(obs_dim, 32)
        self.layer_2 = nn.Linear(32, 32)
        self.layer_3 = nn.Linear(32, num_actions)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))

        return self.layer_3(x)


def main(
    seed: int = 2025,
    env_name: str = "CartPole-v1",
    lr: float = 3e-4,
    replay_buffer_size: int = 1000,
    batch_size: int = 64,
    total_steps: int = 40_000,
    discount_rate: float = 0.99,
    training_frequency: int = 10,
    target_update_frequency: int = 100,
    epsilon_decay: float = 0.995,
    epsilon_min: float = 0.1,
):
    env = gym.make(env_name)

    observation_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    dqn = DQN(observation_dim, num_actions)

    target_dqn = DQN(observation_dim, num_actions)
    target_dqn.load_state_dict(dqn.state_dict())

    optimiser = optim.AdamW(dqn.parameters(), lr=lr)

    replay_buffer = ReplayBuffer(
        buffer_size=replay_buffer_size,
        observation_space=env.observation_space,
        action_space=env.action_space,
        handle_timeout_termination=False,
    )

    epsilon = 1
    current_episode_return = 0
    episode_returns = np.zeros(100)
    episode_num = 0

    obs, info = env.reset()

    for step in range(total_steps):
        if random.random() < epsilon:
            action = env.action_space.sample()
            epsilon = max(epsilon_decay * epsilon, 0.1)

        else:
            with torch.no_grad():
                q_values = dqn(torch.from_numpy(obs))
                action = torch.argmax(q_values).numpy()

        next_obs, reward, terminated, truncated, info = env.step(action)

        replay_buffer.add(obs, next_obs, action, reward, terminated, info)

        done = terminated or truncated
        obs = next_obs

        if step % training_frequency == 0 and replay_buffer.size() > batch_size:
            sample = replay_buffer.sample(batch_size)

            with torch.no_grad():
                target_value, _ = target_dqn(sample.next_observations).max(dim=1)

                target_q_values = sample.rewards + discount_rate * (
                    1 - sample.dones
                ) * target_value.unsqueeze(dim=1)

            q_values = dqn(sample.observations).gather(1, sample.actions)

            loss = F.mse_loss(q_values, target_q_values)

            optimiser.zero_grad()
            loss.backward()

            optimiser.step()

        if step % target_update_frequency == 0:
            target_dqn.load_state_dict(dqn.state_dict())

        if done:
            episode_returns[episode_num % 100] = current_episode_return
            episode_num += 1
            current_episode_return = 0
            obs, info = env.reset()
        else:
            current_episode_return += reward

    print("Number of Episodes", episode_num)
    print("Average Returns from last 100 completed episodes", np.mean(episode_returns))


if __name__ == "__main__":
    tyro.cli(main)
