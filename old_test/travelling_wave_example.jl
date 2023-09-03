using ..FiniteVolumeMethod
include("test_setup.jl")
using Test
using CairoMakie
using OrdinaryDiffEq
using LinearSolve
using ReferenceTests
using StatsBase

## Step 1: Define the mesh 
a, b, c, d, Nx, Ny = 0.0, 3.0, 0.0, 40.0, 60, 80
tri = triangulate_rectangle(a, b, c, d, Nx, Ny; single_boundary=false)
mesh = FVMGeometry(tri)
points = get_points(tri)

## Step 2: Define the boundary conditions 
a₁ = ((x, y, t, u::T, p) where {T}) -> one(T)
a₂ = ((x, y, t, u::T, p) where {T}) -> zero(T)
a₃ = ((x, y, t, u::T, p) where {T}) -> zero(T)
a₄ = ((x, y, t, u::T, p) where {T}) -> zero(T)
bc_fncs = (a₁, a₂, a₃, a₄)
types = (:D, :N, :D, :N)
BCs = BoundaryConditions(mesh, bc_fncs, types)

## Step 3: Define the actual PDE  
f = ((x::T, y::T) where {T}) -> zero(T)
diff_fnc = (x, y, t, u, p) -> p * u
reac_fnc = (x, y, t, u, p) -> p * u * (1 - u)
D, λ = 0.9, 0.99
diff_parameters = D
reac_parameters = λ
final_time = 50.0
u₀ = [f(x, y) for (x, y) in points]
prob = FVMProblem(mesh, BCs; diffusion_function=diff_fnc, reaction_function=reac_fnc,
    diffusion_parameters=diff_parameters, reaction_parameters=reac_parameters,
    initial_condition=u₀, final_time)

## Step 4: Solve
alg = TRBDF2(linsolve=KLUFactorization())
sol = solve(prob, alg; saveat=0.5)

## Step 5: Visualisation 
fig = Figure(resolution=(3024.72f0, 686.64f0), fontsize=38)
for (i, j) in zip(1:3, (1, 51, 101))
    ax = Axis(fig[1, i], width=600, height=600)
    tricontourf!(ax, tri, sol.u[j], levels=0:0.05:1, colormap=:matter)
    tightlimits!(ax)
end
fig

## Step 6: Put the solutions into a matrix format and test the x-invariance by comparing each column to the middle
u_mat = [reshape(u, (Nx, Ny)) for u in sol.u]
all_errs = zeros(length(sol))
err_cache = zeros((Nx - 1) * Ny)
for i in eachindex(sol)
    u = u_mat[i]
    ctr = 1
    for j in union(1:((Nx÷2)-1), ((Nx÷2)+1):Nx)
        for k in 1:Ny
            err_cache[ctr] = 100abs(u[j, k] .- u[Nx÷2, k])
            ctr += 1
        end
    end
    all_errs[i] = mean(err_cache)
end
@test all(all_errs .< 0.05)

## Step 7: Now compare to the exact (travelling wave) solution 
large_time_idx = findfirst(sol.t .== 10)
c = sqrt(λ / (2D))
cₘᵢₙ = sqrt(λ * D / 2)
zᶜ = 0.0
exact_solution = ((z::T) where {T}) -> ifelse(z ≤ zᶜ, 1 - exp(cₘᵢₙ * (z - zᶜ)), zero(T))
err_cache = zeros(Nx * Ny)
all_errs = zeros(length(sol) - large_time_idx + 1)
for (i, t_idx) in pairs(large_time_idx:lastindex(sol))
    u = u_mat[t_idx]
    τ = sol.t[t_idx]
    ctr = 1
    for j in 1:Nx
        for k in 1:Ny
            y = c + (k - 1) * (d - c) / (Ny - 1)
            z = y - c * τ
            exact_wave = exact_solution(z)
            err_cache[ctr] = abs(u[j, k] - exact_wave)
            ctr += 1
        end
    end
    all_errs[i] = mean(err_cache)
end
@test all(all_errs .< 0.1)

## Step 8: Visualise the comparison with the travelling wave
travelling_wave_values = zeros(Ny, length(sol) - large_time_idx + 1)
z_vals = zeros(Ny, length(sol) - large_time_idx + 1)
for (i, t_idx) in pairs(large_time_idx:lastindex(sol))
    u = u_mat[t_idx]
    τ = sol.t[t_idx]
    for k in 1:Ny
        y = c + (k - 1) * (d - c) / (Ny - 1)
        z = y - c * τ
        z_vals[k, i] = z
        travelling_wave_values[k, i] = u[Nx÷2, k]
    end
end
exact_z_vals = collect(LinRange(extrema(z_vals)..., 500))
exact_travelling_wave_values = exact_solution.(exact_z_vals)

ax = Axis(fig[1, 4], width=900, height=600)
colors = cgrad(:matter, length(sol) - large_time_idx + 1; categorical=false)
[lines!(ax, z_vals[:, i], travelling_wave_values[:, i], color=colors[i], linewidth=2) for i in 1:(length(sol)-large_time_idx+1)]
lines!(ax, exact_z_vals, exact_travelling_wave_values, color=:red, linewidth=4, linestyle=:dash)
@test_reference "../docs/src/figures/travelling_wave_problem_test.png" fig
