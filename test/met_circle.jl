using ..FiniteVolumeMethod
using DelaunayTriangulation
using CairoMakie
using OrdinaryDiffEq
using SteadyStateDiffEq
using Test
using LinearAlgebra
using ReferenceTests
using StatsBase
using StableRNGs

include("test_setup.jl")
R = 1.0
θ = collect(LinRange(0, 2π, 100))
θ[end] = θ[begin]
x = R .* cos.(θ)
y = R .* sin.(θ)
boundary_nodes, points = convert_boundary_points_to_indices(x, y)
rng = StableRNG(998877)
tri = triangulate(points; boundary_nodes, rng)
A = get_total_area(tri)
refine!(tri; max_area=1e-3A, min_angle=33.0, rng)
mesh = FVMGeometry(tri)
bc = (x, y, t, u, p) -> zero(u)
type = :D
BCs = BoundaryConditions(mesh, bc, type)
D = 2.5e-5
initial_guess = ones(num_points(tri))
initial_guess[get_boundary_nodes(tri)] .= 0.0
diffusion_function = (x, y, t, u, D) -> D
diffusion_parameters = D
reaction = (x, y, t, u, p) -> one(u)
final_time = Inf # doesn't matter in this case, but we still need to provide it
prob = FVMProblem(mesh, BCs;
    diffusion_function,
    diffusion_parameters,
    reaction_function=reaction,
    final_time=final_time,
    initial_condition=initial_guess,
    steady=true)

# Solve 
alg = DynamicSS(Rosenbrock23())
sol = solve(prob, alg)

# Visualise 
fig = Figure(fontsize=38)
ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"y")
msh = tricontourf!(tri, sol.u, levels = 0:500:10000)
tightlimits!(ax)
Colorbar(fig[1, 2], msh)
fig
@test_reference "../docs/src/figures/circle_mean_exit_time.png" fig

# Test 
exact = [(R^2 - norm(p)^2) / (4D) for p in each_point(tri)]
error_fnc = (fvm_sol, exact_sol) -> 100abs(fvm_sol - exact_sol) / maximum(exact)
@test median(error_fnc.(exact, sol)) < 0.2