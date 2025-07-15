import matplotlib.pyplot as plt
import numpy as np
import scipy


def dydt(_t, y):
    return -0.5 * y


def solution(t):
    p = -0.5 * t
    return np.e**p


sol = scipy.integrate.solve_ivp(dydt, (0.0, 10.0), np.array([1.0]))
actual_y = []

for t in sol.t:
    actual_y.append(solution(t))

plt.plot(sol.t, sol.y[0], label="Scipy")
plt.plot(sol.t, actual_y, label="Solution")
plt.legend()
plt.show()
