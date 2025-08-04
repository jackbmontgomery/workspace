import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

# ------------------------ Hyperparameters ------------------------

BATCH_SIZE = 64
INPUT_DIM = BATCH_SIZE
HIDDEN_DIM = 32
OUTPUT_DIM = BATCH_SIZE
LEARNING_RATE = 3e-4
EPOCHS = 1_000
TAU = 0.2  # For soft update
SEED = 2025

# ------------------------ Model Definition ------------------------


class NN(eqx.Module):
    layer_1: eqx.nn.Linear
    layer_2: eqx.nn.Linear
    layer_3: eqx.nn.Linear

    def __init__(self, key, in_dim, hidden_dim, out_dim):
        key1, key2, key3 = jax.random.split(key, 3)
        self.layer_1 = eqx.nn.Linear(in_dim, hidden_dim, key=key1)
        self.layer_2 = eqx.nn.Linear(hidden_dim, hidden_dim, key=key2)
        self.layer_3 = eqx.nn.Linear(hidden_dim, out_dim, key=key3)

    def __call__(self, x):
        x = jax.nn.sigmoid(self.layer_1(x))
        x = jax.nn.sigmoid(self.layer_2(x))
        return self.layer_3(x)


# ------------------------ Loss & Training Step ------------------------


def loss_fn(model, x, y):
    pred_y = model(x)
    return jnp.mean((y - pred_y) ** 2)


grad_loss_fn = eqx.filter_value_and_grad(loss_fn)


@eqx.filter_jit
def make_step(model, opt_state, x, y, optimizer):
    loss, grads = grad_loss_fn(model, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


# ------------------------ Main Training Loop ------------------------


def main():
    key = jax.random.key(SEED)

    # Generate training data
    x_samples = jnp.linspace(0, 2 * jnp.pi, num=BATCH_SIZE)
    y_samples = jnp.sin(2 * x_samples)

    # Model and optimizer
    model_key, init_key = jax.random.split(key)
    model = NN(model_key, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)

    init_y_pred = model(x_samples)

    optimizer = optax.adamw(LEARNING_RATE)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Training loop
    loss_history = []
    for epoch in range(EPOCHS):
        model, opt_state, loss = make_step(
            model, opt_state, x_samples, y_samples, optimizer
        )
        loss_history.append(loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.8f}")

    # Plot results
    y_pred = model(x_samples)

    plt.plot(x_samples, y_samples, label="Actual")
    plt.plot(x_samples, y_pred, label="Final Model")
    plt.plot(x_samples, init_y_pred, label="Initial Model")
    plt.legend()
    plt.title("Fitting sin(2x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
