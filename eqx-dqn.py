import equinox as eqx
import flashbax as fbx
import gymnax
import jax
import jax.numpy as jnp
import optax
import tyro

jax.config.update("jax_enable_x64", True)


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


def make_buffer(replay_buffer_size, batch_size, env, env_params, key):
    buffer = fbx.make_item_buffer(
        max_length=replay_buffer_size,
        min_length=batch_size,
        sample_batch_size=batch_size,
    )

    obs, env_state = env.reset(key, env_params)
    action = env.action_space(env_params).sample(key)
    next_obs, env_state, reward, done, _ = env.step(key, env_state, action, env_params)

    dummy_experience = {
        "obs": obs,
        "next_obs": next_obs,
        "action": action,
        "reward": reward,
        "terminated": done,
    }

    buffer_state = buffer.init(dummy_experience)
    return buffer, buffer_state


def main(
    seed: int = 2025,
    env_name: str = "CartPole-v1",
    lr: float = 5e-3,
    replay_buffer_size: int = 1000,
    batch_size: int = 64,
    total_steps: int = 5_000,
    discount_rate: float = 0.99,
    training_frequency: int = 10,
    target_update_frequency: int = 100,
    epsilon_decay: float = 0.995,
    epsilon_min: float = 0.1,
    episode_window_size: int = 10,
):
    key = jax.random.key(seed)

    env, env_params = gymnax.make(env_name)
    obs_dim = env.observation_space(env_params).shape[0]
    n_actions = env.action_space(env_params).n

    dqn = DQN(obs_dim, n_actions, key)
    target_dqn = DQN(obs_dim, n_actions, key)

    optimiser = optax.adamw(lr)
    optimiser_state = optimiser.init(eqx.filter(dqn, eqx.is_array))

    buffer, buffer_state = make_buffer(
        replay_buffer_size, batch_size, env, env_params, key
    )

    obs, env_state = env.reset(key, env_params)

    episode_returns = jnp.zeros(episode_window_size)
    episode_num = 1
    episode_return = 0
    eps_samples = jax.random.uniform(key, total_steps)
    epsilon = 1

    for step in range(total_steps):
        if eps_samples[step] < epsilon:
            act_key, key = jax.random.split(key)
            action = env.action_space(env_params).sample(act_key)
            epsilon = jnp.maximum(epsilon * epsilon_decay, epsilon_min)
        else:
            action = jnp.argmax(dqn(obs))

        key_step, key = jax.random.split(key)

        next_obs, env_state, reward, done, _ = env.step(
            key_step, env_state, action, env_params
        )

        terminated = done
        buffer_state = buffer.add(
            buffer_state,
            {
                "obs": obs,
                "next_obs": next_obs,
                "action": action,
                "reward": reward,
                "terminated": terminated,
            },
        )

        episode_return += reward

        if done:
            reset_key, key = jax.random.split(key)

            episode_returns = episode_returns.at[episode_num % episode_window_size].set(
                episode_return
            )

            obs, env_state = env.reset(reset_key, env_params)
            episode_return = 0
            episode_num += 1

            if episode_num % episode_window_size == 0:
                print(
                    f"Episode Number {episode_num} - Average Last 10: {jnp.average(episode_returns):.3f}"
                )

        else:
            obs = next_obs

        if step % training_frequency and buffer.can_sample(buffer_state):
            key_sample, key = jax.random.split(key)
            sample = buffer.sample(buffer_state, key_sample)

            batch_obs = sample.experience["obs"]
            batch_next_obs = sample.experience["next_obs"]
            batch_actions = sample.experience["action"]
            batch_rewards = sample.experience["reward"]
            batch_terminations = sample.experience["terminated"]

            target_q_values = batch_rewards + (
                1 - batch_terminations
            ) * discount_rate * eqx.filter_vmap(target_dqn)(batch_next_obs).max(axis=1)

            @eqx.filter_grad
            def train_step(dqn, target_q_values):
                q_values = eqx.filter_vmap(dqn)(batch_obs)
                q_values = q_values[
                    jnp.arange(q_values.shape[0]), batch_actions.squeeze()
                ]
                return jnp.mean(jnp.power(q_values - target_q_values, 2))

            grads = train_step(dqn, target_q_values)
            updates, optimiser_state = optimiser.update(grads, optimiser_state, dqn)
            dqn = optax.apply_updates(dqn, updates)

        if step % target_update_frequency == 0:
            target_dqn = optax.incremental_update(dqn, target_dqn, 1)


if __name__ == "__main__":
    tyro.cli(main)
