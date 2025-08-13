import chex
import equinox as eqx
import gymnax
import jax
import jax.numpy as jnp
import tyro


@chex.dataclass
class Carry:
    obs: chex.Array
    env_state: chex.ArrayTree


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
    bias = b_init(b_key, (shape[0],))  # bias shape = out_features
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

        carry = Carry(obs=obs, env_state=env_state)

        return carry, step_data

    init_carry = Carry(obs=obs, env_state=env_state)

    prng_keys = jax.random.split(key, num_steps)

    epsilon_values = linear_schedule_array(
        init_eps, min_eps, duration, global_step, num_steps
    )

    xs = StepInput(prng_key=prng_keys, epsilon=epsilon_values)

    final_carry, rollout = jax.lax.scan(rollout_scan_func, init_carry, xs)

    return rollout


def main():
    seed = 2025
    num_envs = 2
    env_id = "CartPole-v1"
    num_steps = 15
    init_eps = 1
    min_eps = 0.05
    exploration_fraction = 0.5
    total_timesteps = 50

    key = jax.random.key(seed)

    env, env_params = gymnax.make(env_id)

    q_network = QNetwork(
        env.observation_space(env_params), env.action_space(env_params), key
    )

    # Multiple Envs
    key_reset = jax.random.split(key, num_envs)
    obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(key_reset, env_params)

    rollout_key = jax.random.split(key, num_envs)

    vmapped_generate_rollout = jax.vmap(
        generate_rollout,
        in_axes=(0, 0, 0, None, None, None, None, None, None, None, None),
    )

    rollout = vmapped_generate_rollout(
        obs,
        env_state,
        rollout_key,
        q_network,
        env,
        env_params,
        num_steps,
        total_timesteps * exploration_fraction,
        0,
        init_eps,
        min_eps,
    )

    print(rollout.done)


if __name__ == "__main__":
    tyro.cli(main)
