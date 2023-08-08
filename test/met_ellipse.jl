using ..FiniteVolumeMethod
using DelaunayTriangulation
using CairoMakie
using OrdinaryDiffEq
using SteadyStateDiffEq
using Test
using LinearAlgebra
using StatsBase
using LinearSolve
using Krylov
using ReferenceTests
using StableRNGs

include("test_setup.jl")
a = 2.0
b = 1.0
θ = LinRange(0, 2π, 100)
x = a * cos.(θ)
y = b * sin.(θ)
x[end] = x[begin]
y[end] = y[begin]
rng = StableRNG(998877)
boundary_nodes, points = convert_boundary_points_to_indices(x, y)
tri = triangulate(points; boundary_nodes, rng)
A = get_total_area(tri)
refine!(tri; max_area=1e-3A/2, min_angle = 33.0, rng)
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
final_time = Inf
prob = FVMProblem(mesh, BCs;
    diffusion_function,
    diffusion_parameters,
    reaction_function=reaction,
    final_time=final_time,
    initial_condition=initial_guess,
    steady=true)
sol = solve(prob, DynamicSS(TRBDF2(linsolve=KrylovJL_GMRES())))
fig = Figure(fontsize=38)
ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"y")
msh = tricontourf!(tri, sol.u, levels = 0:500:16000)
tightlimits!(ax)
Colorbar(fig[1, 2], msh)
fig
resize_to_layout!(fig)
@test_reference "../docs/src/figures/ellipse_mean_exit_time.png" fig
exact = [a^2 * b^2 / (2 * D * (a^2 + b^2)) * (1 - x^2 / a^2 - y^2 / b^2) for (x, y) in each_point(tri)]
error_fnc = (fvm_sol, exact_sol) -> 100abs(fvm_sol - exact_sol) / maximum(exact)
@test median(error_fnc.(exact, sol)) < 0.5
