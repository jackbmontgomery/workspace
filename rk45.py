import matplotlib.pyplot as plt
import numpy as np

SAFETY = 0.9


def dydt(_t, y):
    return -0.5 * y


def solution(t):
    p = -0.5 * t
    return np.e**p


y0 = 1.0
t0 = 0
tf = 10.0
h0 = 0.5


rtol = 1e-3
atol = 1e-6
error_exponent = 1 / 5

y = y0
t = t0
h = h0

ts = []
ys = []
hs = []

ys.append(y)
ts.append(t)

k1 = dydt(t, y)

while t < tf:
    k2 = dydt(t + h / 5, y + h * k1 / 5)
    k3 = dydt(t + 3 * h / 10, y + 3 * h * k1 / 40 + 9 * h * k2 / 40)
    k4 = dydt(t + 4 * h / 5, y + 44 * h * k1 / 45 - 56 * h * k2 / 15 + 32 * h * k3 / 9)
    k5 = dydt(
        t + 8 * h / 9,
        y
        + 19372 * h * k1 / 6561
        - 25360 * h * k2 / 2187
        + 64448 * h * k3 / 6561
        - 212 * h * k4 / 729,
    )
    k6 = dydt(
        t + h,
        y
        + 9017 * h * k1 / 3168
        - 355 * h * k2 / 33
        + 46732 * h * k3 / 5247
        + 49 * h * k4 / 176
        - 5103 * h * k5 / 18656,
    )
    y_next = (
        y
        + 35 * h * k1 / 384
        + 500 * h * k3 / 1113
        + 125 * h * k4 / 192
        - 2187 * h * k5 / 6784
        + 11 * h * k6 / 84,
    )[0]

    k7 = dydt(t + h, y_next)

    z_next = (
        y
        + 5179 * h * k1 / 576000
        + 7571 * h * k3 / 16695
        + 393 * h * k4 / 640
        - 92097 * h * k5 / 339200
        + 187 * h * k6 / 2100
        + h * k7 / 40
    )

    scale = atol + max(np.abs(y), np.abs(y_next)) * rtol
    error = np.abs((y_next - z_next) / scale)

    factor = SAFETY * (1 / error) ** error_exponent
    h_next = factor * h

    if error < 1:
        print(t, "Accepted", "Step Size", h)

        y = y_next
        t += h

        ys.append(y)
        ts.append(t)
        hs.append(h)

        k1 = k7

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
