using Random;
using Plots;

function lorenz_dynamics(sigma::Float64=10.0, beta::Float64=8 / 3, rho::Float64=28.0)::Function
    function rhs(coords::Vector{Float64})
        @assert length(coords) == 3 "coords must be a 3-element vector"

        x, y, z = coords
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z

        return [dx, dy, dz]
    end
    return rhs
end

function rk4_stepper(rhs::Function, dt::Float64=0.1)::Function
    function stepper(coords::Vector{Float64})
        k1 = rhs(coords)
        k2 = rhs(coords .+ dt / 2 .* k1)
        k3 = rhs(coords .+ dt / 2 .* k2)
        k4 = rhs(coords .+ dt .* k3)

        next_coords = coords .+ dt / 6 .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
        return next_coords
    end
    return stepper
end

function rollout_wrapper(stepper::Function, T::Int64)::Function
    function rollout(init_coords::Vector{Float64})
        D = length(init_coords)

        trajectory = Matrix{Float64}(undef, D, T + 1)
        trajectory[:, 1] = init_coords

        coords = init_coords

        for t in 1:T
            coords = stepper(coords)
            trajectory[:, t+1] = coords
        end

        return trajectory
    end
    return rollout
end

lorenz_rhs = lorenz_dynamics()
stepper = rk4_stepper(lorenz_rhs)

T = 1000000
n_trajectories = 100

rollout_fn = rollout_wrapper(stepper, T)

base_coords = fill(10.0, 3)
initial_conditions = [base_coords .+ 0.01 .* randn(3) for _ in 1:n_trajectories]

@time begin
    rollout_fn.(initial_conditions)
end
