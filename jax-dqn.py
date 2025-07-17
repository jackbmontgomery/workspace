import equinox as eqx
import flashbax as fbx
import gymnax
import jax
import jax.numpy as jnp
import optax

import wandb

# ------------------------ Config & Hyperparameters ------------------------

jax.config.update("jax_enable_x64", True)

SEED = 2024
LEARNING_RATE = 3e-4
DISCOUNT = 0.99
EPSILON_DECAY = 0.999
MIN_EPSILON = 0.1
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 1000
REPLAY_BUFFER_MIN = 100
TOTAL_EPISODES = 1000
TARGET_UPDATE_FREQ = 100
TRAIN_FREQ = 10
TAU = 0.005


# ------------------------ DQN Model ------------------------


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


# ------------------------ Loss and Updates ------------------------


def mse_loss(dqn, obs, actions, target_q):
    q_values = jax.vmap(dqn)(obs)
    batch_idx = jnp.arange(q_values.shape[0])
    chosen_q = q_values[batch_idx, actions]
    return jnp.mean((chosen_q - target_q) ** 2)


grad_loss_fn = eqx.filter_value_and_grad(mse_loss)


def soft_update(target, source, tau):
    return jax.tree_map(lambda t, s: (1 - tau) * t + tau * s, target, source)


# ------------------------ Main Training ------------------------


def main():
    # Keys
    key = jax.random.key(SEED)
    key_env, key_model, key_buffer = jax.random.split(key, 3)

    # Environment
    env, env_params = gymnax.make("CartPole-v1")
    obs_dim = env.observation_space(env_params).shape[0]
    n_actions = env.action_space(env_params).n

    # Models
    dqn = DQN(obs_dim, n_actions, key_model)
    target_dqn = DQN(obs_dim, n_actions, key_model)

    # Optimizer
    optimizer = optax.adamw(LEARNING_RATE)
    opt_state = optimizer.init(eqx.filter(dqn, eqx.is_array))

    # Replay buffer
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

    total_steps = 0
    epsilon = 1.0

    wandb.init()

    for episode in range(TOTAL_EPISODES):
        key_env, key_reset, key_episode = jax.random.split(key_env, 3)
        obs, env_state = env.reset(key_reset, env_params)
        done = False
        episode_reward = 0.0

        while not done:
            total_steps += 1

            # Action selection (epsilon-greedy)
            key_env, key_action = jax.random.split(key_env)
            if jax.random.uniform(key_action) < epsilon:
                action = env.action_space(env_params).sample(key_action)
                epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
            else:
                action = jnp.argmax(dqn(obs))

            # Environment step
            next_obs, env_state, reward, done, _ = env.step(
                key_episode, env_state, action, env_params
            )

            # Store transition
            buffer_state = buffer.add(
                buffer_state,
                {
                    "obs": obs,
                    "next_obs": next_obs,
                    "action": action,
                    "reward": reward,
                    "done": done,
                },
            )

            obs = next_obs
            episode_reward += reward

            # Training step
            if total_steps % TRAIN_FREQ == 0 and buffer.can_sample(buffer_state):
                batch = buffer.sample(buffer_state, key_episode)

                batch_obs = batch.experience["obs"]
                batch_next_obs = batch.experience["next_obs"]
                actions = batch.experience["action"]
                rewards = batch.experience["reward"]
                dones = batch.experience["done"]

                # Target computation
                next_q_values = jax.vmap(target_dqn)(batch_next_obs)
                max_next_q = jnp.max(next_q_values, axis=1)
                target_q = rewards + DISCOUNT * (1.0 - dones) * max_next_q

                # Loss and update
                loss, grads = grad_loss_fn(dqn, batch_obs, actions, target_q)
                updates, opt_state = optimizer.update(grads, opt_state, dqn)
                dqn = eqx.apply_updates(dqn, updates)

            # Target update
            if total_steps % TARGET_UPDATE_FREQ == 0:
                target_dqn = soft_update(dqn, target_dqn, TAU)

        wandb.log({"episode_reward": episode_reward, "epsilon": epsilon})

    wandb.finish()


if __name__ == "__main__":
    main()
