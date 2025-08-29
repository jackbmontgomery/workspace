import chex
import equinox as eqx
import gymnax
import jax
import jax.numpy as jnp
import optax

jax.config.update("jax_enable_x64", True)


@chex.dataclass
class StepInput:
    prng_key: chex.PRNGKey
    epsilon: chex.Array


@chex.dataclass
class Carry:
    obs: chex.Array
    env_state: chex.ArrayTree
    q_network: chex.ArrayTree
    done: chex.ArrayTree


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


class Linear(eqx.nn.Linear):
    weight: chex.Array
    bias: chex.Array
    in_features: int
    out_features: int
    use_bias: bool

    def __init__(
        self, in_features, out_features, *, key, std=jnp.sqrt(2), bias_const=0.0
    ):
        super().__init__(in_features, out_features, use_bias=True, key=key)

        self.weight, self.bias = layer_init(
            key, (out_features, in_features), std, bias_const
        )


class QNetwork(eqx.Module):
    layer_1: eqx.nn.Linear
    layer_2: eqx.nn.Linear
    layer_3: eqx.nn.Linear
    norm_1: eqx.nn.LayerNorm
    norm_2: eqx.nn.LayerNorm

    def __init__(self, single_observation_space, single_action_space, key):
        obs_size = jnp.array(single_observation_space.shape).prod()
        num_actions = single_action_space.n

        keys = jax.random.split(key, 3)

        self.layer_1 = Linear(obs_size, 120, key=keys[0])
        self.norm_1 = eqx.nn.LayerNorm(120)

        self.layer_2 = Linear(120, 84, key=keys[1])
        self.norm_2 = eqx.nn.LayerNorm(84)

        self.layer_3 = Linear(84, num_actions, key=keys[2])

    def __call__(self, x):
        x = jax.nn.relu(self.norm_1(self.layer_1(x)))
        x = jax.nn.relu(self.norm_2(self.layer_2(x)))
        return self.layer_3(x)


def linear_schedule_array(
    start_e: float, end_e: float, duration: int, t0: int, num_steps: int
):
    slope = (end_e - start_e) / duration
    t = jnp.arange(t0, t0 + num_steps)
    epsilons = slope * t + start_e
    return jnp.maximum(epsilons, end_e)


class EnvironmentStepper:
    def __init__(self, env_params, env):
        self.env_params = env_params
        self.env = env

    def __call__(self, carry: Carry, step_input: StepInput):
        prev_obs = carry.obs
        prev_done = carry.done

        env_state = carry.env_state
        q_network = carry.q_network

        prng_key = step_input.prng_key
        epsilon = step_input.epsilon

        q_values = q_network(prev_obs)

        max_action = jnp.argmax(q_values, axis=-1)

        random_action = jax.random.randint(
            prng_key,
            shape=max_action.shape,
            minval=0,
            maxval=self.env.action_space(self.env_params).n,
        )

        explore = jax.random.uniform(prng_key) < epsilon

        action = jnp.where(explore, random_action, max_action)
        value = q_values[action]

        obs, env_state, reward, done, _ = self.env.step(
            prng_key, env_state, action, self.env_params
        )

        step_data = StepData(
            obs=prev_obs, action=action, reward=reward, done=prev_done, value=value
        )

        carry = Carry(obs=obs, env_state=env_state, done=done, q_network=q_network)

        return carry, step_data


def generate_rollout_wrapper(num_steps, duration, init_eps, min_eps):
    def rollout_wrapper(
        key: chex.PRNGKey,
        stepper: EnvironmentStepper,
        q_network: QNetwork,
        global_step: int,
    ):
        obs, env_state = stepper.env.reset(key)

        init_carry = Carry(
            obs=obs, env_state=env_state, done=False, q_network=q_network
        )

        prng_keys = jax.random.split(key, num_steps)

        epsilon_values = linear_schedule_array(
            init_eps, min_eps, duration, global_step, num_steps
        )

        step_inputs = StepInput(prng_key=prng_keys, epsilon=epsilon_values)

        return jax.lax.scan(stepper, init_carry, step_inputs)

    return rollout_wrapper


def compute_returns_wrapper(num_steps, gamma, q_lambda):
    def compute_returns(rollout: StepData, final_carry: Carry, q_network: QNetwork):
        returns = jnp.zeros_like(rollout.reward)

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_value = jnp.max(
                    eqx.filter_vmap(q_network)(final_carry.obs), axis=-1
                )
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

        return returns

    return compute_returns


def train_batch_wrapper(env, env_params, update_epochs, batch_size, minibatch_size, tx):
    def train_batch(
        q_network: QNetwork,
        optim_state: chex.ArrayTree,
        rollout: StepData,
        returns: jax.Array,
        key: chex.PRNGKey,
    ):
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

                @eqx.filter_jit
                @eqx.filter_value_and_grad
                def train_step(q_network, target_q_values):
                    q_values = eqx.filter_vmap(q_network)(b_obs[mb_inds])
                    q_values = q_values[
                        jnp.arange(q_values.shape[0]), b_actions[mb_inds]
                    ]

                    return jnp.mean(jnp.power(q_values - target_q_values, 2))

                loss, grads = train_step(q_network, b_returns[mb_inds])

                updates, optim_state = tx.update(grads, optim_state, q_network)
                q_network = eqx.apply_updates(q_network, updates)
        return q_network, optim_state

    return train_batch


def main(
    seed: int = 2025,
    num_envs: int = 8,
    env_id: str = "CartPole-v1",
    lr: float = 5e-3,
    num_steps: int = 128,
    init_eps: float = 1.0,
    min_eps: float = 0.1,
    total_timesteps: int = 50_000,
    exploration_fraction: float = 0.5,
    num_minibatches: int = 4,
    update_epochs: int = 4,
    gamma: float = 0.99,
    q_lambda: float = 0.65,
    max_grad_norm: float = 10.0,
):
    batch_size = int(num_envs * num_steps)
    minibatch_size = int(batch_size // num_minibatches)
    num_iterations = total_timesteps // batch_size

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

    stepper = EnvironmentStepper(env_params=env_params, env=env)

    generate_rollout = generate_rollout_wrapper(
        num_steps,
        total_timesteps * exploration_fraction,
        init_eps,
        min_eps,
    )
    vmap_generate_rollout = jax.vmap(generate_rollout, in_axes=(0, None, None, None))

    compute_returns = compute_returns_wrapper(num_steps, gamma, q_lambda)

    train_batch = train_batch_wrapper(
        env, env_params, update_epochs, batch_size, minibatch_size, tx
    )

    global_step = 0

    for i in range(1, num_iterations + 1):
        iteration_key, key = jax.random.split(key)
        rollout_keys = jax.random.split(key, num_envs)

        final_carry, rollout = vmap_generate_rollout(
            rollout_keys, stepper, q_network, global_step
        )

        global_step += num_envs * num_steps

        returns = compute_returns(rollout, final_carry, q_network)

        q_network, optim_state = train_batch(
            q_network, optim_state, rollout, returns, iteration_key
        )

        total_dones = int(jnp.sum(rollout.done))

        print(
            f"[Iteration {i}/{num_iterations}] "
            f"Global Step: {global_step:,} | "
            f"Dones in rollout: {total_dones}/{num_envs * num_steps}"
        )


if __name__ == "__main__":
    main()
