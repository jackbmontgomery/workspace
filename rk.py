from enum import StrEnum
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy

SAFETY = 0.9

MIN_FACTOR = 0.2
MAX_FACTOR = 10


def rms_norm(x):
    return np.linalg.norm(x) / x.size**0.5


class VanderPoloscillator:
    def __init__(self, mu: float = 2.0):
        self.mu = mu

    def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
        return np.array([self.mu * (y[0] - (y[0] ** 3) / 3 - y[1]), y[0] / self.mu])


class SolverStatus(StrEnum):
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"


class RungeKutta:
    C: np.ndarray = NotImplemented
    A: np.ndarray = NotImplemented
    B: np.ndarray = NotImplemented
    E: np.ndarray = NotImplemented

    order: int = NotImplemented
    error_estimator_order: int = NotImplemented
    n_stages: int = NotImplemented

    status: Optional[SolverStatus] = None

    def __init__(
        self,
        dydt: Callable,
        t0: float,
        tf: float,
        max_step=np.inf,
        rtol: float = 1e-3,
        atol: float = 1e-6,
        first_step_size: Optional[float] = None,
    ):
        self.t = t0
        self.t0 = t0
        self.tf = tf
        self.atol = atol
        self.rtol = rtol
        self.dydt = dydt
        self.error_exponent = -1 / (self.error_estimator_order + 1)
        self.h = first_step_size

    def set_iv(self, y0: np.ndarray):
        self.y = y0
        self.coords_dim = y0.size
        self.f = self.dydt(self.t, self.y)
        self.K = np.empty((self.n_stages + 1, self.coords_dim), dtype=self.y.dtype)

        if self.h is None:
            self.h = self._select_initial_step()

        self.status = "running"

    def _estimate_error(self, K, h):
        return np.dot(K.T, self.E) * h

    def _estimate_error_norm(self, K, h, scale):
        return rms_norm(self._estimate_error(K, h) / scale)

    def _select_initial_step(self):
        interval_length = abs(self.tf - self.t0)
        if interval_length == 0.0:
            return 0.0

        scale = self.atol + np.abs(self.y) * self.rtol
        d0 = rms_norm(self.y / scale)
        d1 = rms_norm(self.f / scale)

        if d0 < 1e-5 or d1 < 1e-5:
            h0 = 1e-6
        else:
            h0 = 0.01 * d0 / d1

        h0 = min(h0, interval_length)
        y1 = self.y + h0 * self.f
        f1 = self.dydt(t0 + h0, y1)
        d2 = rms_norm((f1 - self.f) / scale) / h0

        if d1 <= 1e-15 and d2 <= 1e-15:
            h1 = max(1e-6, h0 * 1e-3)
        else:
            h1 = (0.01 / max(d1, d2)) ** (1 / (self.order + 1))

        return min(100 * h0, h1, interval_length)

    def _rk_step(self):
        self.K[0] = self.f

        for s, (a, c) in enumerate(zip(self.A[1:], self.C[1:]), start=1):
            dy = np.dot(self.K[:s].T, a[:s]) * self.h
            self.K[s] = self.dydt(self.t + c * self.h, self.y + dy)

        y_new = self.y + self.h * np.dot(self.K[:-1].T, self.B)
        f_new = self.dydt(self.t + self.h, y_new)

        self.K[-1] = f_new

        return y_new, f_new

    def step(self):
        step_accepted = False

        while not step_accepted:
            y_new, f_new = self._rk_step()

            scale = self.atol + np.maximum(np.abs(self.y), np.abs(y_new)) * self.rtol
            error_norm = self._estimate_error_norm(self.K, self.h, scale)

            factor = min(MAX_FACTOR, SAFETY * error_norm**self.error_exponent)

            self.h *= factor
            step_accepted = error_norm < 1

        self.t += self.h
        self.y = y_new
        self.f = f_new

        if self.t >= self.tf:
            self.status = "finished"


class RK23(RungeKutta):
    order = 3
    error_estimator_order = 2
    n_stages = 3
    C = np.array([0, 1 / 2, 3 / 4])
    A = np.array([[0, 0, 0], [1 / 2, 0, 0], [0, 3 / 4, 0]])
    B = np.array([2 / 9, 1 / 3, 4 / 9])
    E = np.array([5 / 72, -1 / 12, -1 / 9, 1 / 8])


class RK45(RungeKutta):
    order = 5
    error_estimator_order = 4
    n_stages = 6
    C = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1])
    A = np.array(
        [
            [0, 0, 0, 0, 0],
            [1 / 5, 0, 0, 0, 0],
            [3 / 40, 9 / 40, 0, 0, 0],
            [44 / 45, -56 / 15, 32 / 9, 0, 0],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0],
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
        ]
    )
    B = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84])
    E = np.array(
        [-71 / 57600, 0, 71 / 16695, -71 / 1920, 17253 / 339200, -22 / 525, 1 / 40]
    )


def solve_ivp(solver: RungeKutta, y0: np.ndarray):
    solver.set_iv(y0)
    ts = [solver.t]
    ys = [y0]

    while solver.status != "finished":
        solver.step()
        ts.append(solver.t)
        ys.append(solver.y)

    ts.append(solver.t)
    ys.append(solver.y)

    return ts, ys


if __name__ == "__main__":
    ode = VanderPoloscillator()
    t0 = 0.0
    tf = 100.0
    solver = RK23(ode, t0, tf)

    y0 = np.array([1.0, 0.0])
    ts, ys = solve_ivp(solver, y0)
    ys = np.array(ys)

    scipy_sol = scipy.integrate.solve_ivp(ode, (t0, tf), y0, method="RK23")

    plt.plot(ys[:, 0], ys[:, 1], label="RK")
    plt.plot(scipy_sol.y[0], scipy_sol.y[1], label="Scipy")
    plt.legend()
    plt.show()
