import copy

import equinox as eqx
import flashbax as fbx
import gymnax
import jax
import jax.numpy as jnp
import optax

jax.config.update("jax_enable_x64", True)

SEED = 2025
BATCH_SIZE = 8
REPLAY_BUFFER_SIZE = 10
REPLAY_BUFFER_MIN = 1


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


key = jax.random.key(SEED)

env, env_params = gymnax.make("CartPole-v1")
key_env, key_buffer = jax.random.split(key, 2)

obs_dim = env.observation_space(env_params).shape[0]
n_actions = env.action_space(env_params).n

dqn = DQN(obs_dim, n_actions, key)
target_dqn = copy.copy(dqn)

optimizer = optax.adamw(3e-4)
opt_state = optimizer.init(eqx.filter(dqn, eqx.is_array))

buffer = fbx.make_item_buffer(
    max_length=REPLAY_BUFFER_SIZE,
    min_length=REPLAY_BUFFER_MIN,
    sample_batch_size=BATCH_SIZE,
)


dummy_experience = {
    "obs": jnp.zeros(obs_dim),
    "next_obs": jnp.zeros(obs_dim),
    "action": jnp.array(0),
    "reward": jnp.array(0.0),
    "done": jnp.array(False),
}

buffer_state = buffer.init(dummy_experience)
obs, env_state = env.reset(key_env, env_params)

done = False

while not done:
    action = env.action_space(env_params).sample(key_env)

    next_obs, env_state, reward, done, info = env.step(
        key_env, env_state, action, env_params
    )

    experience = {
        "obs": obs,
        "next_obs": next_obs,
        "action": action,
        "reward": reward,
        "done": done,
    }

    buffer_state = buffer.add(buffer_state, experience)
    obs = next_obs


def loss(dqn, batch_obs, batch_actions, target_q_values):
    q_values = jax.vmap(dqn)(batch_obs)
    batch_idx = jnp.arange(q_values.shape[0])
    chosen_q_values = q_values[batch_idx, batch_actions]
    return jnp.mean((chosen_q_values - target_q_values) ** 2)


grad_loss_fn = eqx.filter_value_and_grad(loss)

batch = buffer.sample(buffer_state, key_buffer)

batch_obs = batch.experience["obs"]
batch_next_obs = batch.experience["next_obs"]
batch_actions = batch.experience["action"]
batch_rewards = batch.experience["reward"]
batch_dones = batch.experience["done"]


for i in range(12):
    next_q_values = jax.vmap(target_dqn)(batch_next_obs)
    max_next_q = jnp.max(next_q_values, axis=1)
    target_q_values = batch_rewards + 0.99 * (1.0 - batch_dones) * max_next_q

    loss_value, grads = grad_loss_fn(dqn, batch_obs, batch_actions, target_q_values)
    updates, opt_state = optimizer.update(grads, opt_state, dqn)
    dqn = optax.apply_updates(dqn, updates)
    print(i, loss_value, "----------")

    if i % 10 == 0 and i > 0:
        target_dqn = copy.deepcopy(dqn)
