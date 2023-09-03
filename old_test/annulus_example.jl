using ..FiniteVolumeMethod
include("test_setup.jl")
using CairoMakie
using LinearSolve
using OrdinaryDiffEq
using ReferenceTests
using Test

## Generate the mesh. 
# When specifying multiple boundary curves, the outer boundary comes first 
# and is given in counter-clockwise order. The inner boundaries then follow. 
R₁ = 0.2
R₂ = 1.0
θ = collect(LinRange(0, 2π, 100))
θ[end] = 0.0 # get the endpoints to match
x = [
    [R₂ .* cos.(θ)],
    [reverse(R₁ .* cos.(θ))]
]
y = [
    [R₂ .* sin.(θ)],
    [reverse(R₁ .* sin.(θ))]
]
boundary_nodes, points = convert_boundary_points_to_indices(x, y)
tri = triangulate(points; boundary_nodes)
A = get_total_area(tri)
refine!(tri;max_area=1e-4A)
mesh = FVMGeometry(tri)

## Define the boundary conditions 
outer_bc = (x, y, t, u, p) -> zero(u)
inner_bc = (x, y, t, u, p) -> 50.0 * (1.0 - exp(-0.5t))
type = [:N, :D]
BCs = BoundaryConditions(mesh, [outer_bc, inner_bc], type)

## Define the problem 
initial_condition_f = (x, y) -> begin
    10 * exp(-25 * ((x + 0.5) * (x + 0.5) + (y + 0.5) * (y + 0.5))) - 5 * exp(-50 * ((x + 0.3) * (x + 0.3) + (y + 0.5) * (y + 0.5))) - 10 * exp(-45 * ((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)))
end
diffusion = (x, y, t, u, p) -> one(u)
points = get_points(tri)
u₀ = [initial_condition_f(x, y) for (x, y) in points]
final_time = 2.0
prob = FVMProblem(mesh, BCs;
    diffusion_function=diffusion,
    final_time=final_time,
    initial_condition=u₀)

## Solve the problem 
alg = TRBDF2(linsolve=KLUFactorization())
sol = solve(prob, alg, parallel=false, saveat=0.2)

## Visualise 
fig = Figure(resolution=(2068.72f0, 686.64f0), fontsize=38)
for (i, j) in zip(1:3, (1, 6, 11))
    ax = Axis(fig[1, i], width=600, height=600)
    tricontourf!(ax, tri, sol.u[j], levels=-10:2:40, colormap=:viridis)
    tightlimits!(ax)
end
fig
@test_reference "../docs/src/figures/annulus_test.png" fig