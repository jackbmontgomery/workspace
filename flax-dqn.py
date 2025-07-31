import random

import flashbax as fbx
import flax.nnx as nnx
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax


class DQN(nnx.Module):
    def __init__(self, input_dim, output_dim, *, rngs: nnx.Rngs):
        k1, k2, k3 = jax.random.split(key, 3)
        self.layer_1 = nnx.Linear(input_dim, 32, rngs=rngs)
        self.layer_2 = nnx.Linear(32, 32, rngs=rngs)
        self.layer_3 = nnx.Linear(32, output_dim, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.layer_1(x))
        x = nnx.relu(self.layer_2(x))
        return self.layer_3(x)


seed = 2025
env_id = "phys2d/CartPole-v1"
max_steps = 5_000
buffer_size = 1000
batch_size = 5
eps_decay = 0.995
eps_min = 0.1
training_frequency = 10
target_update_frequency = 100
discount_rate = 0.99
learning_rate = 5e-3

key = jax.random.key(seed)
rngs = nnx.Rngs(key)

env = gym.make(env_id)

obs_dim = env.observation_space.shape[0]
num_actions = env.action_space.n

replay_buffer = fbx.make_item_buffer(
    max_length=buffer_size, min_length=batch_size, sample_batch_size=batch_size
)

buffer_state = replay_buffer.init(
    {
        "obs": jnp.zeros(obs_dim),
        "next_obs": jnp.zeros(obs_dim),
        "action": jnp.array(0),
        "reward": jnp.array(0.0),
        "terminated": jnp.array(False),
    }
)


dqn = DQN(obs_dim, num_actions, rngs=rngs)
target_dqn = DQN(obs_dim, num_actions, rngs=rngs)
target_dqn.eval()
nnx.update(target_dqn, nnx.state(dqn))

optimiser = nnx.optimizer.Optimizer(dqn, optax.adamw(learning_rate), wrt=nnx.Param)

eps = 1
current_step = 0
episode_num = 0
episode_returns = []

obs, info = env.reset()
done = False
episode_return = 0

while current_step < max_steps:
    current_step += 1
    if random.random() < eps:
        action = env.action_space.sample()
        eps = max(eps * eps_decay, eps_min)
    else:
        action = dqn(obs).argmax()

    next_obs, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated
    episode_return += reward

    buffer_state = replay_buffer.add(
        buffer_state,
        {
            "obs": obs,
            "next_obs": next_obs,
            "action": action,
            "reward": reward,
            "terminated": terminated,
        },
    )

    if current_step % training_frequency:
        key_sample, key = jax.random.split(key)
        sample = replay_buffer.sample(buffer_state, key_sample)

        batch_obs = sample.experience["obs"]
        batch_next_obs = sample.experience["next_obs"]
        batch_actions = sample.experience["action"]
        batch_rewards = sample.experience["reward"]
        batch_terminations = sample.experience["terminated"]

        target_q_values = batch_rewards + (
            1 - batch_terminations
        ) * discount_rate * target_dqn(batch_next_obs).max(axis=1)

        @nnx.grad
        def train_step(dqn: DQN, target_q_values):
            q_values = dqn(batch_obs)
            q_values = q_values[jnp.arange(q_values.shape[0]), batch_actions.squeeze()]
            return jnp.mean(jnp.power(q_values - target_q_values, 2))

        grads = train_step(dqn, target_q_values)
        optimiser.update(grads)

    if current_step % target_update_frequency == 0:
        nnx.update(target_dqn, nnx.state(dqn))

    if done:
        if episode_num % 10 == 0:
            print(
                f"Episode Number {episode_num} - Average Last 10: {np.average(episode_returns):.3f}"
            )
            episode_returns = []

        obs, info = env.reset()
        episode_return = 0
        episode_num += 1

    else:
        obs = next_obs
        episode_returns.append(episode_return)
