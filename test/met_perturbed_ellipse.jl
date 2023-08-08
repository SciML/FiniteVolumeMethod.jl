using ..FiniteVolumeMethod
using DelaunayTriangulation
using CairoMakie
using OrdinaryDiffEq
using SteadyStateDiffEq
using Test
using LinearSolve
using Krylov
using ReferenceTests
using StableRNGs
include("test_setup.jl")
a = 2.0
b = 1.0
θ = LinRange(0, 2π, 100)
ε = 1 / 20
g = θ -> sin(3θ) + cos(5θ) - sin(θ)
h = θ -> cos(3θ) + sin(5θ) - cos(θ)
x = a * (1 .+ ε .* g.(θ)) .* cos.(θ)
y = b * (1 .+ ε .* h.(θ)) .* sin.(θ)
x[end] = x[begin]
y[end] = y[begin]
rng = StableRNG(998877)
boundary_nodes, points = convert_boundary_points_to_indices(x, y)
tri = triangulate(points; boundary_nodes, rng)
A = get_total_area(tri)
refine!(tri; max_area=1e-4A/2, min_angle = 33.0, rng)
mesh = FVMGeometry(tri)
bc = (x, y, t, u, p) -> zero(u)
type = :D
BCs = BoundaryConditions(mesh, bc, type)
D = 2.5e-5
initial_guess = zeros(num_points(tri))
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
ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"y", width = 900, height = 600)
msh = tricontourf!(tri, sol.u, levels = 0:500:16000)
tightlimits!(ax)
Colorbar(fig[1, 2], msh)
resize_to_layout!(fig)
fig
@test_reference "../docs/src/figures/perturbed_ellipse_mean_exit_time.png" fig
