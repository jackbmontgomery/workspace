import copy

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


def dqn_loss(dqn, obs, actions, target_q):
    q_values = jax.vmap(dqn)(obs)
    batch_idx = jnp.arange(q_values.shape[0])
    chosen_q = q_values[batch_idx, actions]
    return jnp.mean((chosen_q - target_q) ** 2)


grad_loss_fn = eqx.filter_value_and_grad(dqn_loss)


def dqn_train_step():
    pass


@chex.dataclass
class Carry:
    key_env: chex.PRNGKey
    env_state: chex.ArrayTree
    dqn: chex.PyTreeDef
    target_dqn: chex.PyTreeDef
    obs: chex.Array
    epsilon: float
    buffer_state: chex.ArrayTree
    optimizer_state: chex.ArrayTree
    total_steps: int
    episode_return: float
    episode_returns: chex.Array
    episode_num: int


def scan_func_wrapper(
    env,
    env_params,
    buffer,
    training_frequency,
    target_update_frequency,
    discount_rate,
    optimizer,
    epsilon_min,
    epsilon_decay,
):
    def scan_func(carry: Carry, _):
        key_env = carry.key_env
        env_state = carry.env_state
        dqn = carry.dqn
        target_dqn = carry.target_dqn
        obs = carry.obs
        epsilon = carry.epsilon
        buffer_state = carry.buffer_state
        optimizer_state = carry.optimizer_state
        total_steps = carry.total_steps
        episode_return = carry.episode_return
        episode_returns = carry.episode_returns
        episode_num = carry.episode_num

        total_steps += 1

        key_env, key_action, key_step, key_reset = jax.random.split(key_env, 4)

        action = jax.lax.cond(
            jax.random.uniform(key_action) < epsilon,
            lambda: env.action_space(env_params).sample(key_action),
            lambda: jnp.argmax(dqn(obs)),
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

        def train_step(args):
            dqn, optimizer_state, key = args

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

            loss, grads = grad_loss_fn(dqn, batch_obs, batch_actions, target_q)
            updates, new_optimizer_state = optimizer.update(grads, optimizer_state, dqn)
            new_dqn = eqx.apply_updates(dqn, updates)

            return new_dqn, new_optimizer_state, key_env

        do_training = (total_steps % training_frequency == 0) & buffer.can_sample(
            buffer_state
        )

        dqn, optimizer_state, key = jax.lax.cond(
            do_training,
            train_step,
            lambda args: args,
            (dqn, optimizer_state, key_env),
        )

        def update_target(args):
            target_dqn, dqn = args
            target_dqn = copy.deepcopy(dqn)
            return target_dqn, dqn

        target_dqn, dqn = jax.lax.cond(
            total_steps % target_update_frequency == 0,
            update_target,
            lambda args: args,
            (target_dqn, dqn),
        )

        obs, env_state = jax.lax.cond(
            done,
            lambda: env.reset(key_reset, env_params),
            lambda: (next_obs, env_state),
        )

        episode_return, episode_returns, episode_num = jax.lax.cond(
            done,
            lambda: (
                0.0,
                episode_returns.at[episode_num % 100].set(episode_return),
                episode_num + 1,
            ),
            lambda: (episode_return + reward, episode_returns, episode_num),
        )

        epsilon = jnp.maximum(epsilon_min, epsilon * epsilon_decay)

        carry = Carry(
            key_env=key_env,
            env_state=env_state,
            dqn=dqn,
            target_dqn=target_dqn,
            obs=obs,
            epsilon=epsilon,
            buffer_state=buffer_state,
            optimizer_state=optimizer_state,
            total_steps=total_steps,
            episode_return=episode_return,
            episode_returns=episode_returns,
            episode_num=episode_num,
        )

        return carry, reward

    return scan_func


def main(
    seed: int = 2025,
    env_name: str = "CartPole-v1",
    lr: float = 3e-4,
    replay_buffer_size: int = 1000,
    replay_buffer_min_size: int = 10,
    batch_size: int = 64,
    total_steps: int = 100_000,
    discount_rate: float = 0.99,
    training_frequency: int = 10,
    target_update_frequency: int = 100,
    epsilon_decay: float = 0.995,
    epsilon_min: float = 0.1,
):
    key = jax.random.key(seed)
    key_env, key_model, key_buffer = jax.random.split(key, 3)

    env, env_params = gymnax.make(env_name)
    obs_dim = env.observation_space(env_params).shape[0]
    n_actions = env.action_space(env_params).n

    dqn = DQN(obs_dim, n_actions, key_model)
    target_dqn = copy.deepcopy(dqn)

    optimizer = optax.adamw(lr)
    optimizer_state = optimizer.init(eqx.filter(dqn, eqx.is_array))

    buffer = fbx.make_item_buffer(
        max_length=replay_buffer_size,
        min_length=replay_buffer_min_size,
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

    scan_func = scan_func_wrapper(
        env,
        env_params,
        buffer,
        training_frequency,
        target_update_frequency,
        discount_rate,
        optimizer,
        epsilon_min,
        epsilon_decay,
    )

    obs, env_state = env.reset(key_env, env_params)

    episode_returns = jnp.zeros(100)

    init_carry = Carry(
        key_env=key_env,
        env_state=env_state,
        dqn=dqn,
        target_dqn=target_dqn,
        obs=obs,
        epsilon=1.0,
        buffer_state=buffer_state,
        optimizer_state=optimizer_state,
        total_steps=0,
        episode_return=0.0,
        episode_returns=episode_returns,
        episode_num=0,
    )

    final_carry, outputs = jax.lax.scan(scan_func, init_carry, None, total_steps)

    print(final_carry.episode_num)
    print(final_carry.episode_returns)


if __name__ == "__main__":
    tyro.cli(main)
