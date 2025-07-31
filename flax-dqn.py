import chex
import flashbax as fbx
import flax.nnx as nnx
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax


class DQN(nnx.Module):
    def __init__(self, input_dim, output_dim, *, rngs: nnx.Rngs):
        self.layer_1 = nnx.Linear(input_dim, 32, rngs=rngs)
        self.layer_2 = nnx.Linear(32, 32, rngs=rngs)
        self.layer_3 = nnx.Linear(32, output_dim, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.layer_1(x))
        x = nnx.relu(self.layer_2(x))
        return self.layer_3(x)


@chex.dataclass
class Carry:
    key: chex.PRNGKey
    dqn: nnx.State
    target_dqn: nnx.State
    obs: chex.Array
    eps: float
    buffer_state: chex.ArrayTree
    optimiser: nnx.Optimizer
    current_step: int
    episode_return: float
    episode_returns: chex.Array
    episode_num: int


def make_training_scan_func(
    env: gym.Env,
    replay_buffer,
    training_frequency: int,
    target_update_frequency: int,
    eps_decay: float,
    eps_min: float,
    discount_rate: float,
    episode_window_size: int,
):
    def training_func(carry: Carry, eps_sample):
        key = carry.key
        dqn = carry.dqn
        optimiser = carry.optimiser
        target_dqn = carry.target_dqn
        eps = carry.eps
        obs = carry.obs
        episode_return = carry.episode_return
        episode_returns = carry.episode_returns
        buffer_state = carry.buffer_state
        episode_num = carry.episode_num

        current_step = carry.current_step + 1

        if eps_sample < eps:
            action = env.action_space.sample()
            eps = max(eps * eps_decay, eps_min)
        else:
            action = dqn(obs).argmax()

        next_obs, reward, terminated, truncated, _infos = env.step(action)

        done = terminated or truncated

        episode_return += reward

        buffer_state = replay_buffer.add(
            buffer_state,
            {
                "obs": obs,
                "next_obs": next_obs,
                "action": action,
                "reward": reward,
                "terminated": terminated,
            },
        )

        if done:
            if episode_num > 0 and episode_num % 10 == 0:
                print(
                    f"Episode Number {episode_num} - Average Last 10: {np.average(episode_returns):.3f}"
                )
            obs, _infos = env.reset()
            episode_return = 0
            episode_num += 1

        else:
            obs = next_obs
            episode_returns = episode_returns.at[episode_num % episode_window_size].set(
                episode_return
            )

        if current_step % training_frequency:
            key_sample, key = jax.random.split(key)
            sample = replay_buffer.sample(buffer_state, key_sample)

            batch_obs = sample.experience["obs"]
            batch_next_obs = sample.experience["next_obs"]
            batch_actions = sample.experience["action"]
            batch_rewards = sample.experience["reward"]
            batch_terminations = sample.experience["terminated"]

            target_q_values = batch_rewards + (
                1 - batch_terminations
            ) * discount_rate * target_dqn(batch_next_obs).max(axis=1)

            @nnx.grad
            def train_step(dqn: nnx.Module, target_q_values):
                q_values = dqn(batch_obs)
                q_values = q_values[
                    jnp.arange(q_values.shape[0]), batch_actions.squeeze()
                ]
                return jnp.mean(jnp.power(q_values - target_q_values, 2))

            grads = train_step(dqn, target_q_values)
            optimiser.update(grads)

        if current_step % target_update_frequency == 0:
            nnx.update(target_dqn, nnx.state(dqn))

        carry = Carry(
            key=key,
            dqn=dqn,
            target_dqn=target_dqn,
            obs=obs,
            eps=eps,
            buffer_state=buffer_state,
            optimiser=optimiser,
            current_step=current_step,
            episode_return=episode_return,
            episode_returns=episode_returns,
            episode_num=episode_num,
        )
        return carry, ()

    return training_func


def main(
    seed: int = 2025,
    env_id: str = "phys2d/CartPole-v1",
    lr: float = 5e-3,
    buffer_size: int = 1000,
    batch_size: int = 64,
    total_steps: int = 5_000,
    discount_rate: float = 0.99,
    training_frequency: int = 10,
    target_update_frequency: int = 100,
    eps_decay: float = 0.995,
    eps_min: float = 0.1,
    episode_window_size: int = 100,
):
    key = jax.random.key(seed)
    rngs = nnx.Rngs(key)

    env = gym.make(env_id)

    obs_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    replay_buffer = fbx.make_item_buffer(
        max_length=buffer_size, min_length=batch_size, sample_batch_size=batch_size
    )

    buffer_state = replay_buffer.init(
        {
            "obs": jnp.zeros(obs_dim),
            "next_obs": jnp.zeros(obs_dim),
            "action": jnp.array(0),
            "reward": jnp.array(0.0),
            "terminated": jnp.array(False),
        }
    )

    dqn = DQN(obs_dim, num_actions, rngs=rngs)
    target_dqn = DQN(obs_dim, num_actions, rngs=rngs)
    target_dqn.eval()
    nnx.update(target_dqn, nnx.state(dqn))

    optimiser = nnx.optimizer.Optimizer(dqn, optax.adamw(lr), wrt=nnx.Param)

    eps = 1
    current_step = 0
    episode_num = 0
    episode_returns = jnp.empty(episode_window_size)
    eps_samples = jax.random.uniform(key, total_steps)

    obs, _infos = env.reset()
    episode_return = 0

    training_scan_fun = make_training_scan_func(
        env,
        replay_buffer,
        training_frequency,
        target_update_frequency,
        eps_decay,
        eps_min,
        discount_rate,
        episode_window_size,
    )

    init_carry = Carry(
        key=key,
        dqn=dqn,
        target_dqn=target_dqn,
        obs=obs,
        eps=eps,
        buffer_state=buffer_state,
        optimiser=optimiser,
        current_step=current_step,
        episode_return=episode_return,
        episode_returns=episode_returns,
        episode_num=episode_num,
    )

    final_carry, _ = jax.lax.scan(training_scan_fun, init_carry, eps_samples)
    print(final_carry.episode_returns)


if __name__ == "__main__":
    main()
