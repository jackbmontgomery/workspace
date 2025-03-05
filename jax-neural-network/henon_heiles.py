import jax.numpy as jnp
import jax
import time
from tqdm import tqdm


def calculate_intial_conditions(H, x_0, y_0, py_0):
    return jnp.array(
        [
            x_0,
            y_0,
            jnp.sqrt(
                2 * H
                - 2 * (x_0**2) * y_0
                + 2 * (y_0**3) / 3
                - (x_0**2)
                - (y_0**2)
                - (py_0**2)
            ),
            py_0,
        ]
    )


@jax.jit
def runge_kutta_4(coords, dt):
    def henon_heiles(t, coords):
        return jnp.array(
            [
                coords[2],
                coords[3],
                -coords[0] * (1 + 2 * coords[1]),
                -(coords[1] + coords[0] ** 2 - coords[1] ** 2),
            ]
        )

    k_1 = dt * henon_heiles(0, coords)
    k_2 = dt * henon_heiles(0, coords + (1 / 2) * k_1)
    k_3 = dt * henon_heiles(0, coords + (1 / 2) * k_2)
    k_4 = dt * henon_heiles(0, coords + k_3)

    return coords + (1 / 6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)


H = 0.125
dt = 0.01

x_0 = 0
y_0 = -0.25
py_0 = 0

coords = calculate_intial_conditions(H, x_0, y_0, py_0)


start_time = time.time()
for _ in tqdm(range(1000000)):
    coords = runge_kutta_4(coords, dt)

end_time = time.time()

print(end_time - start_time)
