import chex
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


@chex.dataclass
class Carry:
    dqn: chex.ArrayTree
    target_dqn: chex.ArrayTree
    epsilon: float
    step: int
    obs: chex.Array
    optimiser_state: chex.ArrayTree
    env_state: chex.ArrayTree
    buffer_state: chex.ArrayTree
    episode_num: int
    episode_return: float
    episode_returns: chex.Array


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


def make_scan_training_func(
    env,
    env_params,
    buffer,
    optimiser,
    episode_window_size,
    training_frequency,
    target_update_frequency,
    discount_rate,
    epsilon_decay,
    epsilon_min,
):
    def training_func(carry: Carry, key_step):
        dqn = carry.dqn
        target_dqn = carry.target_dqn
        obs = carry.obs
        env_state = carry.env_state

        optimiser_state = carry.optimiser_state
        env_state = carry.env_state
        buffer_state = carry.buffer_state

        episode_num = carry.episode_num
        episode_return = carry.episode_return
        episode_returns = carry.episode_returns

        epsilon = carry.epsilon
        step = carry.step + 1

        def explore(eps, key):
            action = env.action_space(env_params).sample(key)
            eps = jnp.maximum(eps * epsilon_decay, epsilon_min)
            return action, eps

        def exploit(eps, key):
            action = jnp.argmax(dqn(obs))
            return action, eps

        action, epsilon = jax.lax.cond(
            jax.random.uniform(key_step) < epsilon,
            explore,
            exploit,
            *(epsilon, key_step),
        )

        next_obs, env_state, reward, done, _ = env.step(
            key_step, env_state, action, env_params
        )

        terminated = jnp.logical_and(
            done, env_state.time < env_params.max_steps_in_episode
        )

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

        def conclude_episode(
            episode_returns, episode_return, episode_num, obs, env_state
        ):
            episode_returns = episode_returns.at[episode_num % episode_window_size].set(
                episode_return
            )

            obs, env_state = env.reset(key_step, env_params)
            episode_return = 0.0
            episode_num += 1
            return episode_returns, episode_return, episode_num, obs, env_state

        def move_obs(episode_returns, episode_return, episode_num, obs, env_state):
            obs = next_obs
            return episode_returns, episode_return, episode_num, obs, env_state

        episode_returns, episode_return, episode_num, obs, env_state = jax.lax.cond(
            done,
            conclude_episode,
            move_obs,
            *(episode_returns, episode_return, episode_num, obs, env_state),
        )

        def train_dqn(model, optimiser_state):
            sample = buffer.sample(buffer_state, key_step)

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

            grads = train_step(model, target_q_values)
            updates, optimiser_state = optimiser.update(grads, optimiser_state, model)
            model = optax.apply_updates(model, updates)
            return model, optimiser_state

        def skip_training_fn(dqn, optimiser_state):
            return dqn, optimiser_state

        dqn, optimiser_state = jax.lax.cond(
            jnp.logical_and(
                step % training_frequency == 0, buffer.can_sample(buffer_state)
            ),
            train_dqn,
            skip_training_fn,
            *(dqn, optimiser_state),
        )
        target_dqn = jax.lax.cond(
            step % target_update_frequency,
            lambda target_dqn: optax.incremental_update(dqn, target_dqn, 1),
            lambda target_dqn: target_dqn,
            target_dqn,
        )

        carry = Carry(
            dqn=dqn,
            target_dqn=target_dqn,
            epsilon=epsilon,
            step=step,
            obs=obs,
            optimiser_state=optimiser_state,
            env_state=env_state,
            buffer_state=buffer_state,
            episode_num=episode_num,
            episode_return=episode_return,
            episode_returns=episode_returns,
        )

        return carry, ()

    return training_func


def main(
    seed: int = 100,
    env_name: str = "CartPole-v1",
    lr: float = 5e-3,
    replay_buffer_size: int = 1000,
    batch_size: int = 64,
    total_steps: int = 100_000,
    discount_rate: float = 0.99,
    training_frequency: int = 10,
    target_update_frequency: int = 100,
    epsilon_decay: float = 0.995,
    epsilon_min: float = 0.1,
    episode_window_size: int = 100,
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
    epsilon = 1.0
    step = 0

    scan_training_func = make_scan_training_func(
        env,
        env_params,
        buffer,
        optimiser,
        episode_window_size,
        training_frequency,
        target_update_frequency,
        discount_rate,
        epsilon_decay,
        epsilon_min,
    )

    init_carry = Carry(
        dqn=dqn,
        target_dqn=target_dqn,
        epsilon=epsilon,
        step=step,
        obs=obs,
        optimiser_state=optimiser_state,
        env_state=env_state,
        buffer_state=buffer_state,
        episode_num=episode_num,
        episode_return=episode_return,
        episode_returns=episode_returns,
    )

    step_keys = jax.random.split(key, total_steps)

    final_carry, _ = jax.lax.scan(scan_training_func, init_carry, step_keys)

    print(jnp.mean(final_carry.episode_returns))


if __name__ == "__main__":
    tyro.cli(main)
