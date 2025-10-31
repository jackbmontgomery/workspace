import matplotlib.pyplot as plt
import numpy as np


def plot_solution(X, Y, fdm_solution, _analytical_solution, elev=30, azim=-45):
    fig = plt.figure(figsize=(12, 5))

    # FDM solution
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    surf1 = ax1.plot_surface(X, Y, fdm_solution, cmap="viridis")
    ax1.set_title("FDM Solution")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Time")
    fig.colorbar(surf1, ax=ax1, shrink=0.6)
    ax1.view_init(elev=elev, azim=azim)

    # Analytical solution
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    surf2 = ax2.plot_surface(X, Y, _analytical_solution, cmap="viridis")
    ax2.set_title("Analytical Solution")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Time")
    fig.colorbar(surf2, ax=ax2, shrink=0.6)
    ax2.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    plt.show()


L = 1
T = 1
Nx = 20
Nt = 200
a = 0.1

dt = T / Nt
dx = L / Nx

F = a * dt / dx**2

xs = np.linspace(0, L, Nx)
ts = np.linspace(0, T, Nt)

# Time by Space
fdm_solution = np.zeros((Nt, Nx))

fdm_solution[0] = np.sin(np.pi * xs)

method = "CN"

if method == "Explicit":
    print(F, "needs to be below 0.5")
    for t in range(len(ts) - 1):
        fdm_solution[t + 1, 1:-1] = fdm_solution[t, 1:-1] + F * (
            fdm_solution[t, 2:] - 2 * fdm_solution[t, 1:-1] + fdm_solution[t, :-2]
        )

elif method == "Implicit":
    A = np.zeros((Nx, Nx))

    for i in range(1, Nx - 1):
        A[i, i] = 2.0 * F + 1
        A[i, i + 1] = -F
        A[i, i - 1] = -F

    A[0, 0] = 1
    A[0, 1] = 0

    A[-1, -2] = 0
    A[-1, -1] = 1
    for t in range(len(ts) - 1):
        fdm_solution[t + 1] = np.linalg.solve(A, fdm_solution[t])

else:
    A = np.eye(Nx, Nx)

    for i in range(1, Nx - 1):
        A[i, i - 1] = -F / 2
        A[i, i] = F + 1
        A[i, i + 1] = -F / 2

    for t in range(len(ts) - 1):
        b = np.zeros((Nx,))
        for i in range(1, Nx - 1):
            b[i] = (
                F * fdm_solution[t, i - 1] / 2
                + (1 - F) * fdm_solution[t, i]
                + F * fdm_solution[t, i + 1] / 2
            )
        fdm_solution[t + 1] = np.linalg.solve(A, b)


def analytical_solution(T, X, D=0.01, L=1.0):
    k = np.pi / L
    U = np.sin(k * X) * np.exp(-D * k**2 * T)
    return U


X, T = np.meshgrid(xs, ts)

plot_solution(X, T, fdm_solution, analytical_solution(T, X, a, L))
