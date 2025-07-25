import manim
import numpy as np
from manim import ThreeDScene
from scipy.integrate import solve_ivp


def lorenz_system(t, state, sigma=10, rho=28, beta=8 / 3):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]


def ode_solution_points(function, state0, time, dt=0.01):
    solution = solve_ivp(
        function, t_span=(0, time), y0=state0, t_eval=np.arange(0, time, dt)
    )
    return solution.y.T


class LorenzAttractor(ThreeDScene):
    # Can use https://docs.manim.community/en/stable/reference/manim.animation.changing.TracedPath.html in future
    def construct(self):
        #  Create axis

        self.set_camera_orientation(
            phi=75 * manim.DEGREES, theta=30 * manim.DEGREES, zoom=0.7
        )

        axes = manim.ThreeDAxes(
            x_range=(-50, 50, 5),
            y_range=(-50, 50, 5),
            z_range=(-0, 50, 5),
        ).shift([0.0, 0.0, -2])
        self.add(axes)

        # Single Trajectory

        num_trajectories = 10
        time = 30

        colors = manim.color_gradient([manim.BLUE_A, manim.BLUE_E], num_trajectories)
        line_graphs = []
        epsilon = 0.001

        for i, color in enumerate(colors):
            initial_conditions = [10, 10, 10 + i * epsilon]

            trajectory = ode_solution_points(lorenz_system, initial_conditions, time)

            line_graph = manim.ShowPassingFlash(
                axes.plot_line_graph(
                    trajectory[:, 0],
                    trajectory[:, 1],
                    trajectory[:, 2],
                    line_color=color,
                    stroke_width=4,
                    add_vertex_dots=False,
                )
            )

            line_graphs.append(line_graph)

        self.begin_ambient_camera_rotation(rate=0.25, about="theta")
        self.play(
            *line_graphs,
            run_time=time,
            rate_func=manim.rate_functions.linear,
        )
        self.stop_ambient_camera_rotation(about="theta")
