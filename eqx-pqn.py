import chex
import equinox as eqx
import gymnax
import jax
import jax.numpy as jnp
import optax
import tyro


@chex.dataclass
class Carry:
    obs: chex.Array
    env_state: chex.ArrayTree
    done: chex.ArrayTree


@chex.dataclass
class StepInput:
    prng_key: chex.PRNGKey
    epsilon: chex.Array


@chex.dataclass
class StepData:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    value: chex.Array


def layer_init(key, shape, std=jnp.sqrt(2), bias_const=0.0):
    w_key, b_key = jax.random.split(key)
    w_init = jax.nn.initializers.orthogonal(std)
    b_init = jax.nn.initializers.constant(bias_const)
    weight = w_init(w_key, shape)
    bias = b_init(b_key, (shape[0],))

    return weight, bias


class Linear(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, in_features, out_features, key, std=jnp.sqrt(2), bias_const=0.0):
        self.weight, self.bias = layer_init(
            key, (out_features, in_features), std, bias_const
        )

    def __call__(self, x):
        return jnp.dot(x, self.weight.T) + self.bias


class QNetwork(eqx.Module):
    layers: list

    def __init__(self, single_observation_space, single_action_space, key):
        obs_size = jnp.array(single_observation_space.shape).prod()
        num_actions = single_action_space.n

        keys = jax.random.split(key, 4)

        self.layers = [
            Linear(obs_size, 120, keys[0]),
            eqx.nn.LayerNorm(120),
            jax.nn.relu,
            Linear(120, 84, keys[1]),
            eqx.nn.LayerNorm(84),
            jax.nn.relu,
            Linear(84, num_actions, keys[2]),
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x) if callable(layer) else x
        return x


def linear_schedule_array(
    start_e: float, end_e: float, duration: int, t0: int, num_steps: int
):
    slope = (end_e - start_e) / duration
    t = jnp.arange(t0, t0 + num_steps)
    epsilons = slope * t + start_e
    return jnp.maximum(epsilons, end_e)


def generate_rollout(
    obs,
    env_state,
    key,
    q_network,
    env,
    env_params,
    num_steps,
    duration,
    global_step,
    init_eps,
    min_eps,
):
    def rollout_scan_func(carry: Carry, step_input: StepInput):
        obs = carry.obs
        env_state = carry.env_state

        prng_key = step_input.prng_key
        epsilon = step_input.epsilon

        q_values = q_network(obs)

        max_action = jnp.argmax(q_values, axis=-1)

        random_action = jax.random.randint(
            prng_key,
            shape=max_action.shape,
            minval=0,
            maxval=env.action_space(env_params).n,
        )

        explore = jax.random.uniform(prng_key) < epsilon

        action = jnp.where(explore, random_action, max_action)

        value = q_values[action]

        obs, env_state, reward, done, _ = env.step(
            prng_key, env_state, action, env_params
        )

        step_data = StepData(
            obs=obs, action=action, reward=reward, done=done, value=value
        )

        def reset(obs, env_state, key_reset=prng_key, env_params=env_params):
            return env.reset(key_reset, env_params)

        obs, env_state = jax.lax.cond(
            done, reset, lambda o, s: (o, s), *(obs, env_state)
        )

        carry = Carry(obs=obs, env_state=env_state, done=done)

        return carry, step_data

    init_carry = Carry(obs=obs, env_state=env_state, done=False)

    prng_keys = jax.random.split(key, num_steps)

    epsilon_values = linear_schedule_array(
        init_eps, min_eps, duration, global_step, num_steps
    )

    xs = StepInput(prng_key=prng_keys, epsilon=epsilon_values)

    return jax.lax.scan(rollout_scan_func, init_carry, xs)


def main(
    seed: int = 2025,
    num_envs: int = 2,
    env_id: str = "CartPole-v1",
    lr: float = 1e-3,
    num_steps: int = 15,
    init_eps: float = 1.0,
    min_eps: float = 0.05,
    exploration_fraction: float = 0.5,
    total_timesteps: int = 50,
    num_minibatches: int = 4,
    update_epochs: int = 4,
    gamma: float = 0.99,
    q_lambda: float = 0.65,
    max_grad_norm: float = 10.0,
):
    batch_size = int(num_envs * num_steps)
    minibatch_size = int(batch_size // num_minibatches)

    key = jax.random.key(seed)

    env, env_params = gymnax.make(env_id)

    q_network = QNetwork(
        env.observation_space(env_params), env.action_space(env_params), key
    )

    tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.radam(learning_rate=lr),
    )

    optim_state = tx.init(eqx.filter(q_network, eqx.is_array))

    vmapped_generate_rollout = jax.vmap(
        generate_rollout,
        in_axes=(0, 0, 0, None, None, None, None, None, None, None, None),
    )

    global_step = 0

    key_reset = jax.random.split(key, num_envs)
    obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(key_reset, env_params)

    rollout_key = jax.random.split(key, num_envs)

    final_carry, rollout = vmapped_generate_rollout(
        obs,
        env_state,
        rollout_key,
        q_network,
        env,
        env_params,
        num_steps,
        total_timesteps * exploration_fraction,
        global_step,
        init_eps,
        min_eps,
    )

    returns = jnp.zeros_like(rollout.reward)

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_value = jnp.max(eqx.filter_vmap(q_network)(final_carry.obs), axis=-1)
            nextnonterminal = 1.0 - final_carry.done
            returns = returns.at[:, t].set(
                rollout.reward[:, t] + gamma * next_value * nextnonterminal
            )
        else:
            nextnonterminal = 1.0 - rollout.done[:, t + 1]
            next_value = rollout.value[:, t + 1]
            returns = returns.at[:, t].set(
                rollout.reward[:, t]
                + gamma
                * (q_lambda * returns[:, t + 1] + (1 - q_lambda) * next_value)
                * nextnonterminal
            )

    b_obs = rollout.obs.reshape((-1,) + env.observation_space(env_params).shape)
    b_actions = rollout.action.reshape((-1,) + env.action_space(env_params).shape)
    b_returns = returns.reshape(-1)

    b_inds = jnp.arange(batch_size)

    for epoch in range(update_epochs):
        permute_key, key = jax.random.split(key)

        b_inds = jax.random.permutation(key, batch_size)

        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]

            @eqx.filter_grad
            def train_step(q_network, target_q_values):
                q_values = eqx.filter_vmap(q_network)(b_obs[mb_inds])
                q_values = q_values[jnp.arange(q_values.shape[0]), b_actions[mb_inds]]
                return jnp.mean(jnp.power(q_values - target_q_values, 2))

            grads = train_step(q_network, b_returns[mb_inds])

            updates, optim_state = tx.update(grads, optim_state, q_network)
            q_network = eqx.apply_updates(q_network, updates)
            # q_network = optax.apply_updates(q_network, updates)


if __name__ == "__main__":
    tyro.cli(main)
