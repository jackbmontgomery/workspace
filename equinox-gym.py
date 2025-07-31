import random

import equinox as eqx
import flashbax as fbx
import gymnasium as gym
import jax
import jax.numpy as jnp
import optax


class DQN(eqx.Module):
    layer_1: eqx.nn.Linear
    layer_2: eqx.nn.Linear
    layer_3: eqx.nn.Linear

    def __init__(self, input_dim, output_dim, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.layer_1 = eqx.nn.Linear(input_dim, 32, key=k1)
        self.layer_2 = eqx.nn.Linear(32, 32, key=k2)
        self.layer_3 = eqx.nn.Linear(32, output_dim, key=k3)

    def __call__(self, x):
        x = jax.nn.relu(self.layer_1(x))
        x = jax.nn.relu(self.layer_2(x))
        return self.layer_3(x)


eqx.filter_jit()


def train_step(
    dqn,
    target_dqn,
    buffer,
    buffer_state,
    optimiser,
    optimiser_state,
    key,
):
    key_env, key_sample = jax.random.split(key)
    batch = buffer.sample(buffer_state, key_sample)

    batch_obs = batch.experience["obs"]
    batch_actions = batch.experience["action"]
    batch_rewards = batch.experience["reward"]
    batch_terminations = batch.experience["terminated"]

    batch_next_obs = batch.experience["next_obs"]

    next_q_values = jax.vmap(target_dqn)(batch_next_obs)
    max_next_q = jnp.max(next_q_values, axis=1)
    target_q = batch_rewards + 0.99 * (1.0 - batch_terminations) * max_next_q

    def loss_fn(model: DQN):
        q_values = jax.vmap(model)(batch_obs)
        q_values = q_values[jnp.arange(q_values.shape[0]), batch_actions.squeeze()]
        return jnp.mean((q_values - target_q) ** 2)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(dqn)
    updates, optimiser_state = optimiser.update(grads, optimiser_state, dqn)
    dqn = eqx.apply_updates(dqn, updates)
    return dqn, optimiser_state, key_env


key = jax.random.key(2025)
learning_rate = 3e-4
replay_buffer_size = 1000
batch_size = 64
training_frequency = 10
target_update_frequency = 100
max_time_steps = 100_000
eps_decay = 0.995
eps_min = 0.1

env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
num_actions = env.action_space.n


dqn = DQN(obs_dim, num_actions, key)
target_dqn = dqn

optimiser = optax.adamw(learning_rate)
optimiser_state = optimiser.init(eqx.filter(dqn, eqx.is_array))

buffer = fbx.make_item_buffer(
    max_length=replay_buffer_size,
    min_length=batch_size,
    sample_batch_size=batch_size,
)

dummy_experience = {
    "obs": jnp.zeros(obs_dim),
    "next_obs": jnp.zeros(obs_dim),
    "action": jnp.array(0),
    "reward": jnp.array(0.0),
    "terminated": jnp.array(False),
}


buffer_state = buffer.init(dummy_experience)
total_time_step = 0
num_episodes = 0
eps = 1

while total_time_step < max_time_steps:
    obs, info = env.reset()
    done = False
    episode_return = 0
    while not done:
        total_time_step += 1
        if eps < random.random():
            action = dqn(jnp.array(obs)).argmax().item()
            eps = max(eps * eps_decay, eps_min)
        else:
            action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)

        experience = {
            "obs": obs,
            "next_obs": next_obs,
            "action": action,
            "reward": reward,
            "terminated": terminated,
        }

        buffer_state = buffer.add(buffer_state, experience)

        obs = next_obs
        done = terminated or truncated
        episode_return += reward

        if total_time_step % training_frequency == 0 & buffer.can_sample(buffer_state):
            dqn, optimiser_state, key = train_step(
                dqn, target_dqn, buffer, buffer_state, optimiser, optimiser_state, key
            )
        if total_time_step % target_update_frequency == 0:
            target_dqn = dqn

    num_episodes += 1
    if num_episodes % 10 == 0:
        print(num_episodes, episode_return)
