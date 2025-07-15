import typing

import numpy as np
import numpy.typing as nptyping


class VanderPoloscillator:
    def __init__(self, mu: float = 2.0):
        self.mu = mu

    def __call__(self, t: float, y: nptyping.ArrayLike) -> nptyping.ArrayLike:
        return np.array([self.mu * (y[0] - (y[0] ** 3) / 3 - y[1]), y[0] / self.mu])


class RK23:
    order = 3
    error_estimator_order = 2
    n_stages = 3
    C = np.array([0, 1 / 2, 3 / 4])
    A = np.array([[0, 0, 0], [1 / 2, 0, 0], [0, 3 / 4, 0]])
    B = np.array([2 / 9, 1 / 3, 4 / 9])
    E = np.array([5 / 72, -1 / 12, -1 / 9, 1 / 8])

    def __init__(
        self,
        dydt: typing.Callable,
        t0: float,
        y0: float,
        t_bound: typing.Tuple[float, float],
        max_step=np.inf,
        rtol: float = 1e-3,
        atol: float = 1e-6,
        first_step=None,
    ):
        self.y_old = None
        self.f = self.fun(self.t, self.y)

        if first_step is None:
            self.h = self._select_initial_step(
                self.fun,
                self.t,
                self.y,
                t_bound,
                max_step,
                self.f,
                self.direction,
                self.error_estimator_order,
                self.rtol,
                self.atol,
            )
        else:
            self.h = first_step

        self.K = np.empty((self.n_stages + 1, self.n), dtype=self.y.dtype)
        self.error_exponent = -1 / (self.error_estimator_order + 1)
        self.h_previous = None


def main():
    system = VanderPoloscillator()
    t0 = 0.0
    y0 = np.array([1.0, 0.0])

    rtol = (1e-3,)
    atol = (1e-6,)

    d0 = np.linalg.norm(y0)
    d1 = np.linalg.norm(system(t0, y0))

    scale = atol + np.abs(y0) * rtol

    if d0 < 10e-5 or d1 < 10e-5:
        h0 = 10e-6
    else:
        h0 = 0.01 * (d0 / d1)

    y1 = y0 + h0 * system(t0, y0)

    d2 = np.linalg.norm(system(t0 + h0, y1) - system(t0, y0)) / h0

    if d1 < 10e-15 or d2 < 10e-15:
        h1 = np.max(10e-6, h0 * 10e-3)
    else:
        h1 = np.power(0.01 / np.max(d1, d2), 1 / 4)

    h = np.min(100 * h0, h1)

    print("Starting Step Size", h)


if __name__ == "__main__":
    main()
