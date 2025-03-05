import jax.numpy as jnp
import jax.nn as nn
import jax
import itertools
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from tqdm import tqdm


def plot_figure(x_values, y_values, resolution=100):
    x1, x2 = x_values[:, 0], x_values[:, 1]
    y_values = y_values.flatten()

    # Create a grid
    x1_lin = jnp.linspace(min(x1), max(x1), resolution)
    x2_lin = jnp.linspace(min(x2), max(x2), resolution)
    X1, X2 = jnp.meshgrid(x1_lin, x2_lin)
    # Interpolate y-values onto the grid
    Y_interp = griddata(x_values, y_values, (X1, X2), method="cubic")

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X1, X2, Y_interp, cmap="viridis", edgecolor="none")

    # Labels
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("Y")

    plt.show()


def func(x1, x2):
    return jnp.sin(3 * x1) * jnp.sin(x2)


num = 10

samples = jnp.linspace(-jnp.pi, jnp.pi, num=num)

x_values = jnp.array(list(itertools.product(samples, samples)))
y_values = jnp.array([func(x1, x2) for x1, x2 in x_values]).reshape(len(x_values), 1)


class AdamOptimizer:
    def __init__(self, params, lr=0.001, beta1=0.8, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {p: jnp.zeros_like(params[p]) for p in params}
        self.v = {p: jnp.zeros_like(params[p]) for p in params}
        self.t = 0

    def step(self, grads):
        self.t += 1
        for p in self.params.keys():
            self.m[p] = self.beta1 * self.m[p] + (1 - self.beta1) * grads[p]
            self.v[p] = self.beta2 * self.v[p] + (1 - self.beta2) * (grads[p] ** 2)

            m_hat = self.m[p] / (1 - self.beta1**self.t)
            v_hat = self.v[p] / (1 - self.beta2**self.t)

            self.params[p] -= self.lr * m_hat / (jnp.sqrt(v_hat) + self.eps)
        return self.params


input_dim = 2
hidden_dim = 32
output_dim = 1

initializer = jax.nn.initializers.glorot_normal()

params = {
    "weights_0": initializer(jax.random.key(1), (input_dim, hidden_dim), jnp.float32),
    "weights_1": initializer(jax.random.key(3), (hidden_dim, output_dim), jnp.float32),
    "bias_0": initializer(jax.random.key(2), (1, hidden_dim), jnp.float32),
    "bias_1": initializer(jax.random.key(4), (1, output_dim), jnp.float32),
}


def forward(params, x):
    weights_0 = params["weights_0"]
    bias_0 = params["bias_0"]
    weights_1 = params["weights_1"]
    bias_1 = params["bias_1"]

    hidden = nn.sigmoid(jnp.matmul(x, weights_0) + bias_0)
    output = jnp.matmul(hidden, weights_1) + bias_1
    return output


def loss(params, x, y):
    output = forward(params, x)
    errors = jnp.power(output - y, 2)
    return jnp.sum(errors)


deriv_loss = jax.jit(jax.grad(loss, argnums=[0]))
# deriv_loss = jax.grad(loss, argnums=[0])

optimizer = AdamOptimizer(params)
losses = []
for i in tqdm(range(25000)):
    grads = deriv_loss(params, x_values, y_values)[0]
    params = optimizer.step(grads)
    if (i + 1) % 10 == 0:
        losses.append(loss(params, x_values, y_values))

print(loss(params, x_values, y_values))

# y_pred = jnp.array([forward(params, x_value) for x_value in x_values]).reshape(
#     len(x_values), 1
# )
# plot_figure(x_values, y_pred)
# plt.plot(losses)
# plt.show()
# for i in tqdm(range(100000)):
#     weights_0_grad, bias_0_grad, weights_1_grad, bias_1_grad = deriv_loss(
#         weights_0, bias_0, weights_1, bias_1, x_values, y_values
#     )

#     weights_0 += step_size * weights_0_grad
#     weights_1 += step_size * weights_1_grad
#     bias_0 += step_size * bias_0_grad
#     bias_1 += step_size * bias_1_grad

# print("loss", loss(weights_0, bias_0, weights_1, bias_1, x_values, y_values))

# y_pred = jnp.array(
#     [forward(x_value, weights_0, bias_0, weights_1, bias_1) for x_value in x_values]
# ).reshape(len(x_values), 1)

# plot_figure(x_values, y_pred)
