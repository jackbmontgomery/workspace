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
    key_env: chex.PRNGKey
    env_state: chex.ArrayTree
    dqn: chex.PyTreeDef
    target_dqn: chex.PyTreeDef
    obs: chex.Array
    epsilon: float
    buffer_state: chex.ArrayTree
    optimiser_state: chex.ArrayTree
    total_steps: int
    episode_return: float
    episode_returns: chex.Array
    episode_num: int


def create_scan_step_fn(
    env,
    env_params,
    buffer,
    training_frequency,
    target_update_frequency,
    discount_rate,
    optimiser,
    epsilon_min,
    epsilon_decay,
    episode_window_size,
):
    def step_fn(carry: Carry, _):
        key_env = carry.key_env
        env_state = carry.env_state
        dqn = carry.dqn
        target_dqn = carry.target_dqn
        obs = carry.obs
        epsilon = carry.epsilon
        buffer_state = carry.buffer_state
        optimiser_state = carry.optimiser_state
        total_steps = carry.total_steps
        episode_return = carry.episode_return
        episode_returns = carry.episode_returns
        episode_num = carry.episode_num

        total_steps += 1

        key_env, key_action, key_step, key_reset = jax.random.split(key_env, 4)

        def explore(epsilon):
            action = env.action_space(env_params).sample(key_action)
            epsilon = jnp.maximum(epsilon * epsilon_decay, epsilon_min)
            return action, epsilon

        def exploit(epsilon):
            action = jnp.argmax(dqn(obs))
            return action, epsilon

        action, epsilon = jax.lax.cond(
            jax.random.uniform(key_action) < epsilon, explore, exploit, epsilon
        )

        next_obs, env_state, reward, done, _ = env.step(
            key_step, env_state, action, env_params
        )

        terminated = jax.lax.cond(
            done & env_state.time < env_params.max_steps_in_episode,
            lambda: True,
            lambda: False,
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

        def train_step(
            dqn,
            optimiser_state,
            key,
        ):
            key_env, key_sample = jax.random.split(key)
            batch = buffer.sample(buffer_state, key_sample)

            batch_obs = batch.experience["obs"]
            batch_next_obs = batch.experience["next_obs"]
            batch_actions = batch.experience["action"]
            batch_rewards = batch.experience["reward"]
            batch_terminations = batch.experience["terminated"]

            next_q_values = jax.vmap(target_dqn)(batch_next_obs)
            max_next_q = jnp.max(next_q_values, axis=1)
            target_q = (
                batch_rewards + discount_rate * (1.0 - batch_terminations) * max_next_q
            )

            def loss_fn(model: DQN):
                q_values = jax.vmap(model)(batch_obs)
                q_values = q_values[
                    jnp.arange(q_values.shape[0]), batch_actions.squeeze()
                ]
                return jnp.mean((q_values - target_q) ** 2)

            loss, grads = eqx.filter_value_and_grad(loss_fn)(dqn)
            updates, optimiser_state = optimiser.update(grads, optimiser_state, dqn)
            dqn = eqx.apply_updates(dqn, updates)
            return dqn, optimiser_state, key_env

        def no_train_step(
            dqn,
            optimiser_state,
            key,
        ):
            return dqn, optimiser_state, key

        dqn, optimiser_state, key = jax.lax.cond(
            (total_steps % training_frequency == 0) & buffer.can_sample(buffer_state),
            train_step,
            no_train_step,
            *(
                dqn,
                optimiser_state,
                key_env,
            ),
        )

        target_dqn = jax.lax.cond(
            total_steps % target_update_frequency == 0,
            lambda: dqn,
            lambda: target_dqn,
        )

        def update_done(
            next_obs, env_state, episode_return, episode_returns, episode_num
        ):
            obs, env_state = env.reset(key_reset, env_params)
            episode_returns = episode_returns.at[episode_num % episode_window_size].set(
                episode_return
            )
            episode_num += 1
            episode_return = 0.0
            return obs, env_state, episode_return, episode_returns, episode_num

        def update_not_done(
            next_obs, env_state, episode_return, episode_returns, episode_num
        ):
            episode_return += reward
            return next_obs, env_state, episode_return, episode_returns, episode_num

        obs, env_state, episode_return, episode_returns, episode_num = jax.lax.cond(
            done,
            update_done,
            update_not_done,
            *(next_obs, env_state, episode_return, episode_returns, episode_num),
        )

        carry = Carry(
            key_env=key_env,
            env_state=env_state,
            dqn=dqn,
            target_dqn=target_dqn,
            obs=obs,
            epsilon=epsilon,
            buffer_state=buffer_state,
            optimiser_state=optimiser_state,
            total_steps=total_steps,
            episode_return=episode_return,
            episode_returns=episode_returns,
            episode_num=episode_num,
        )

        return carry, ()

    return step_fn


def main(
    seed: int = 2025,
    env_name: str = "CartPole-v1",
    lr: float = 3e-4,
    replay_buffer_size: int = 1000,
    batch_size: int = 64,
    total_steps: int = 100_000,
    discount_rate: float = 0.99,
    training_frequency: int = 10,
    target_update_frequency: int = 100,
    epsilon_decay: float = 0.995,
    epsilon_min: float = 0.05,
    episode_window_size: int = 100,
):
    key = jax.random.key(seed)
    key_env, key_model, key_buffer = jax.random.split(key, 3)

    env, env_params = gymnax.make(env_name)
    obs_dim = env.observation_space(env_params).shape[0]
    n_actions = env.action_space(env_params).n

    dqn = DQN(obs_dim, n_actions, key_model)
    target_dqn = dqn

    optimiser = optax.adamw(lr)
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

    env_step_fn = create_scan_step_fn(
        env,
        env_params,
        buffer,
        training_frequency,
        target_update_frequency,
        discount_rate,
        optimiser,
        epsilon_min,
        epsilon_decay,
        episode_window_size,
    )

    obs, env_state = env.reset(key_env, env_params)

    episode_returns = jnp.zeros(episode_window_size)

    init_carry = Carry(
        key_env=key_env,
        env_state=env_state,
        dqn=dqn,
        target_dqn=target_dqn,
        obs=obs,
        epsilon=1.0,
        buffer_state=buffer_state,
        optimiser_state=optimiser_state,
        total_steps=0,
        episode_return=0.0,
        episode_returns=episode_returns,
        episode_num=0,
    )

    final_carry, _ = jax.lax.scan(env_step_fn, init_carry, None, total_steps)

    print("Number of Episodes", final_carry.episode_num)
    print(
        "Average Returns from last 100 completed episodes",
        jnp.mean(final_carry.episode_returns),
    )


if __name__ == "__main__":
    tyro.cli(main)
