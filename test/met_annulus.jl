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
R₁ = 2.0
R₂ = 3.0
θ = (collect ∘ LinRange)(0, 2π, 250)
θ[end] = θ[begin]
x = [
    [R₂ .* cos.(θ)],
    [reverse(R₁ .* cos.(θ))] # inner boundaries are clockwise
]
y = [
    [R₂ .* sin.(θ)],
    [reverse(R₁ .* sin.(θ))] # inner boundaries are clockwise
]
boundary_nodes, points = convert_boundary_points_to_indices(x, y)
rng = StableRNG(1234)
tri = triangulate(points; boundary_nodes, rng=rng)
A = get_total_area(tri)
refine!(tri; max_area = 1e-4A, rng)
mesh = FVMGeometry(tri)
outer_bc = (x, y, t, u, p) -> zero(u)
inner_bc = (x, y, t, u, p) -> zero(u)
type = [:D, :N]
BCs = BoundaryConditions(mesh, [outer_bc, inner_bc], type)
D = 6.25e-4
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
ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"y", width=600, height=300)
pt_mat = get_points(tri)
T_mat = [T[j] for T in each_triangle(tri), j in 1:3]
msh = mesh!(ax, pt_mat, T_mat, color=sol, colorrange=(0, 900))
Colorbar(fig[1, 2], msh)
resize_to_layout!(fig)
@test_reference "../docs/src/figures/annulus_mean_exit_time.png" fig
exact = [(R₂^2 - norm(p)^2) / (4D) + R₁^2 * log(norm(p) / R₂) / (2D) for p in each_point(tri)]
error_fnc = (fvm_sol, exact_sol) -> 100abs(fvm_sol - exact_sol) / maximum(exact)
@test median(error_fnc.(exact, sol)) < 0.1
