import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.buffers import ReplayBuffer
from torch import nn, optim

import wandb


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


env_name = "CartPole-v1"
learning_rate = 3e-4
num_episodes = 50
all_episode_rewards = []
batch_size = 64
eps_decay = 0.9999
buffer_size = 1000
gamma = 0.99

env = gym.make(env_name)

observation_dim = env.observation_space.shape[0]
num_actions = env.action_space.n

dqn = DQN(observation_dim, num_actions)

target_dqn = DQN(observation_dim, num_actions)
target_dqn.load_state_dict(dqn.state_dict())

optimiser = optim.AdamW(dqn.parameters(), lr=learning_rate)

replay_buffer = ReplayBuffer(
    buffer_size=buffer_size,
    observation_space=env.observation_space,
    action_space=env.action_space,
    handle_timeout_termination=False,
)

total_steps = 0
eps = 1

wandb.init(
    project="dqn",
    config={
        "env": env_name,
        "lr": learning_rate,
        "buffer_size": buffer_size,
        "batch_size": batch_size,
        "eps_decay": eps_decay,
        "gamma": gamma,
    },
)

for i, episode in enumerate(range(num_episodes)):
    done = False
    episode_rewards = 0

    obs, info = env.reset()

    while not done:
        total_steps += 1

        if random.random() < eps:
            action = env.action_space.sample()
            eps = max(eps_decay * eps, 0.1)

        else:
            with torch.no_grad():
                q_values = dqn(torch.from_numpy(obs))
                action = torch.argmax(q_values).numpy()

        next_obs, reward, terminated, truncated, info = env.step(action)

        replay_buffer.add(obs, next_obs, action, reward, terminated, info)

        done = terminated or truncated
        obs = next_obs
        episode_rewards += reward

        if total_steps % 10 == 0 and replay_buffer.size() > batch_size:
            sample = replay_buffer.sample(batch_size)

            with torch.no_grad():
                target_value, _ = target_dqn(sample.next_observations).max(dim=1)

                target_q_values = sample.rewards + gamma * (
                    1 - sample.dones
                ) * target_value.unsqueeze(dim=1)

            q_values = dqn(sample.observations).gather(1, sample.actions)

            loss = F.mse_loss(q_values, target_q_values)

            optimiser.zero_grad()
            loss.backward()

            optimiser.step()

        if total_steps % 100 == 0:
            target_dqn.load_state_dict(dqn.state_dict())

    wandb.log({"episode_rewards": episode_rewards, "eps": eps})
    all_episode_rewards.append(episode_rewards)

    if i % 1000 == 0:
        print(i, np.average(all_episode_rewards[-100:]))

env.close()

video_env = RecordVideo(
    gym.make(env_name, render_mode="rgb_array"),
    video_folder="videos",
    episode_trigger=lambda episode_id: True,
)

num_eval_episodes = 5

for ep in range(num_eval_episodes):
    obs, info = video_env.reset()
    done = False
    episode_rewards = 0

    while not done:
        with torch.no_grad():
            q_values = dqn(torch.from_numpy(obs))
            action = torch.argmax(q_values).numpy()

        obs, reward, terminated, truncated, info = video_env.step(action)
        done = terminated or truncated
        episode_rewards += reward

    print(f"Evaluation Episode {ep + 1} Episode Reward: {episode_rewards}")

video_env.close()
