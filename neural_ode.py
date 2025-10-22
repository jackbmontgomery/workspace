import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from diffrax import ODETerm, PIDController, SaveAt, Tsit5, diffeqsolve
from tqdm import tqdm

PLOT = True


class Func(eqx.Module):
    out_scale: jax.Array
    mlp: eqx.nn.MLP

    def __init__(self, data_size, width_size, depth, key):
        super().__init__()
        self.out_scale = jnp.array(1.0)
        self.mlp = eqx.nn.MLP(
            in_size=data_size,
            out_size=data_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            final_activation=jnn.tanh,
            key=key,
        )

    def __call__(self, t, y, args):
        return self.out_scale * self.mlp(y)


class NeuralODE(eqx.Module):
    func: Func

    def __init__(self, data_size, width_size, depth, key):
        super().__init__()
        self.func = Func(data_size, width_size, depth, key)

    def __call__(self, ts, y0):
        solution = diffeqsolve(
            ODETerm(self.func),
            Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            stepsize_controller=PIDController(rtol=1e-3, atol=1e-6),
            saveat=SaveAt(ts=ts),
        )
        return solution.ys


def oscillator(t, y, args):
    x, z = y
    k, m, c = args
    x_dot = z
    z_dot = -(c * z + k * x) / m
    return jnp.array([x_dot, z_dot])


@eqx.filter_value_and_grad
def grad_loss(model, ts, yi):
    y_pred = model(ts, yi[0])
    return jnp.mean((yi - y_pred) ** 2)


@eqx.filter_jit
def make_step(ts, yi, model, opt_state):
    loss, grads = grad_loss(model, ts, yi)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


key = jax.random.key(2025)

model = NeuralODE(2, 64, 2, key)

terms = ODETerm(oscillator)
solver = Tsit5()

y0 = jnp.array([1.0, 1.0])
args = (2.0, 1.0, 0.5)
t0, t1 = 0.0, 10.0
ts = jnp.linspace(t0, t1, 1000)
lr = 3e-3

solution = diffeqsolve(
    terms, solver, t0=t0, t1=t1, dt0=0.1, y0=y0, args=args, saveat=SaveAt(ts=ts)
)
optim = optax.adabelief(lr)

opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

for i in tqdm(range(5000)):
    loss, model, opt_state = make_step(ts, solution.ys, model, opt_state)


prediction = model(ts, y0)

if PLOT:
    plt.plot(solution.ts, prediction[:, 0], label="Prediction x (position)")
    # plt.plot(solution.ts, prediction[:, 1], label="Prediction z (velocity)")

    plt.plot(solution.ts, solution.ys[:, 0], label="Actual x (position)")
    # plt.plot(solution.ts, solution.ys[:, 1], label="Actual z (velocity)")
    plt.xlabel("t")
    plt.legend()
    plt.show()
