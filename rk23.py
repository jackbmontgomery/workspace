import matplotlib.pyplot as plt
import numpy as np

SAFETY = 0.9


def exponential_decay(_t, y):
    return -0.5 * y


def solution(t):
    p = -0.5 * t
    return np.e**p


y0 = 1.0
t0 = 0
tf = 10.0
h0 = 0.5


rtol = 1e-4
atol = 1e-6
error_exponent = 1 / 3

y = y0
t = t0
h = h0

ts = []
ys = []
hs = []

ys.append(y)
ts.append(t)

k1 = exponential_decay(t, y)

while t < tf:
    k2 = exponential_decay(t + (1 / 2) * h, y + (1 / 2) * h * k1)
    k3 = exponential_decay(t + (3 / 4) * h, y + (0) * h * k1 + (3 / 4) * h * k2)

    y_next = y + (2 / 9) * k1 * h + (1 / 3) * k2 * h + (4 / 9) * k3 * h

    k4 = exponential_decay(t + h, y_next)

    z_next = y + h * ((7 / 24) * k1 + (1 / 4) * k2 + (1 / 3) * k3 + (1 / 8) * k4)

    scale = atol + np.maximum(np.abs(y), np.abs(y_next)) * rtol
    error = np.abs((y_next - z_next) / scale)

    factor = SAFETY * h * (1 / error) ** error_exponent
    h_next = factor

    if error < 1:
        print(t, "Accepted", "Step Size", h)

        y = y_next
        t += h

        hs.append(h)
        ys.append(y)
        ts.append(t)

        k1 = k4

    else:
        print(t, "Rejected", h)

    h = h_next

real_ys = []

for t in ts:
    real_ys.append(solution(t))

print("Num Steps", len(ts))
print("Average Step size", np.average(hs))
plt.plot(ts, ys, label="rk45")
plt.plot(ts, real_ys, label="solution")
plt.legend()
plt.show()
