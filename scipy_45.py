import matplotlib.pyplot as plt
import numpy as np
import scipy


class VanderPoloscillator:
    def __init__(self, mu: float = 2.0):
        self.mu = mu

    def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
        return np.array([self.mu * (y[0] - (y[0] ** 3) / 3 - y[1]), y[0] / self.mu])


ode = VanderPoloscillator()
t0 = 0.0
tf = 1000.0
sol = scipy.integrate.solve_ivp(ode, (t0, tf), np.array([1.0, 0.0]), method="RK23")
print(len(sol.t))
plt.plot(sol.y[:, 0], sol.y[:, 1])
plt.show()
