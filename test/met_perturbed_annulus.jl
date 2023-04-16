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

## Define the multiply-connected geometry
R₁ = 2.0
R₂ = 3.0
g₁ = θ -> sin(3θ) + cos(5θ)
g₂ = θ -> cos(3θ)
θ = (collect ∘ LinRange)(0, 2π, 250)
θ[end] = θ[begin] # close the boundary properly
ε = 1 / 20
inner_R = R₁ .* (1 .+ ε .* g₁.(θ))
outer_R = R₂ .* (1 .+ ε .* g₂.(θ))
x = [
    [outer_R .* cos.(θ)],
    [reverse(inner_R .* cos.(θ))] # inner boundaries are clockwise
]
y = [
    [outer_R .* sin.(θ)],
    [reverse(inner_R .* sin.(θ))] # inner boundaries are clockwise
]
boundary_nodes, points = convert_boundary_points_to_indices(x, y)
rng = StableRNG(1234)
tri = triangulate(points; boundary_nodes, rng)
A = get_total_area(tri)
refine!(tri; max_area=1e-4A, rng)

## Define the FVMProblem
mesh = FVMGeometry(tri)
outer_bc = (x, y, t, u, p) -> zero(u)
inner_bc = (x, y, t, u, p) -> zero(u)
type = [:D, :N] # absorbing outer, reflecting inner
BCs = BoundaryConditions(mesh, [outer_bc, inner_bc], type)
D = 6.25e-4
initial_guess = zeros(num_points(tri)) # start the initial guess at the zero vector
diffusion_function = (x, y, t, u, D) -> D
diffusion_parameters = D
reaction = (x, y, t, u, p) -> one(u)
final_time = Inf # doesn't actually get used, but we still need to provide it
prob = FVMProblem(mesh, BCs;
    diffusion_function,
    diffusion_parameters,
    reaction_function=reaction,
    final_time=final_time,
    initial_condition=initial_guess,
    steady=true)

## Now solve - we use a SteadyState solver with a Krylov method
sol = solve(prob, DynamicSS(TRBDF2(linsolve=KrylovJL_GMRES())), parallel=true)

## Visualise
fig = Figure(fontsize=38)
ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"y", width=600, height=300)
pt_mat = get_points(tri)
T_mat = [T[j] for T in each_triangle(tri), j in 1:3]
msh = mesh!(ax, pt_mat, T_mat, color=sol, colorrange=(0, 900))
Colorbar(fig[1, 2], msh)
resize_to_layout!(fig)
@test_reference "../docs/src/figures/perturbed_annulus_mean_exit_time.png" fig