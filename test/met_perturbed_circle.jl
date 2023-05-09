using ..FiniteVolumeMethod
using DelaunayTriangulation
using CairoMakie
using OrdinaryDiffEq
using SteadyStateDiffEq
using LinearSolve
using LinearAlgebra
using ReferenceTests
using StableRNGs

include("test_setup.jl")
R = 1.0
θ = LinRange(0, 2π, 100)
ε = 1 / 20
g = θ -> sin(3θ) + cos(5θ) - sin(θ)
R_bnd = 1 .+ ε .* g.(θ)
x = R_bnd .* cos.(θ)
y = R_bnd .* sin.(θ)
x[end] = x[begin]
y[end] = y[begin] # ensure circular array 
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
initial_guess = [(R^2 - norm(p)^2) / (4D) for p in each_point(tri)]
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
alg = DynamicSS(TRBDF2(linsolve=KrylovJL_GMRES()))
sol = solve(prob, alg, parallel=true)
fig = Figure(fontsize=38)
ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"y")
pt_mat = get_points(tri)
T_mat = [T[j] for T in each_triangle(tri), j in 1:3]
msh = mesh!(ax, pt_mat, T_mat, color=sol, colorrange=(0, 10000))
Colorbar(fig[1, 2], msh)
@test_reference "../docs/src/figures/perturbed_circle_mean_exit_time.png" fig
