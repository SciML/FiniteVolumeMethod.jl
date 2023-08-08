using ..FiniteVolumeMethod
include("test_setup.jl")
using Test
using CairoMakie
using OrdinaryDiffEq
using LinearSolve
using StatsBase
using StableRNGs
using ReferenceTests
using Bessels
using ElasticArrays

## Step 1: Generate the mesh 
r = LinRange(1, 1, 100)
θ = LinRange(0, 2π, 100)
x = @. r * cos(θ)
y = @. r * sin(θ)
x[end] = x[begin];
y[end] = y[begin]; # make sure the curve connects at the endpoints
boundary_nodes, points = convert_boundary_points_to_indices(x, y; existing_points=ElasticMatrix{Float64}(undef, 2, 0))
rng = StableRNG(191919)
tri = triangulate(points; boundary_nodes, rng)
A = get_total_area(tri)
refine!(tri; max_area=1e-4A, rng)
mesh = FVMGeometry(tri)
points = get_points(tri)

## Step 2: Define the boundary conditions 
bc = (x, y, t, u, p) -> u
types = :dudt
BCs = BoundaryConditions(mesh, bc, types)

## Step 3: Define the actual PDE  
f = (x, y) -> sqrt(besseli(0.0, sqrt(2) * sqrt(x^2 + y^2)))
D = (x, y, t, u, p) -> u
R = (x, y, t, u, p) -> u * (1 - u)
points = get_points(tri)
u₀ = [f(points[:, i]...) for i in axes(points, 2)]
final_time = 0.10
prob = FVMProblem(mesh, BCs; diffusion_function=D, reaction_function=R, initial_condition=u₀, final_time)

## Step 4: Solve
alg = FBDF(linsolve=UMFPACKFactorization(), autodiff = false)
sol = solve(prob, alg; saveat=0.025)

## Step 5: Visualisation 
fig = Figure(resolution=(2068.72f0, 686.64f0), fontsize=38)
for (i, j) in zip(1:3, (1, 3, 5))
    ax = Axis(fig[1, i], width=600, height=600)
    tricontourf!(ax, tri, sol.u[j], levels=1:0.01:1.4, colormap=:matter)
    tightlimits!(ax)
end
fig
@test_reference "../docs/src/figures/reaction_diffusion_test.png" fig

## Step 6: Define the exact solution for comparison later 
function reaction_diffusion_exact_solution(x, y, t)
    u_exact = zeros(length(x))
    for i in eachindex(x)
        u_exact[i] = exp(t) * sqrt(besseli(0.0, sqrt(2) * sqrt(x[i]^2 + y[i]^2)))
    end
    return u_exact
end

## Step 7: Compare the results
all_errs = [Float64[] for _ in eachindex(sol)]
u_exact = [reaction_diffusion_exact_solution(points[1, :], points[2, :], τ) for τ in sol.t]
u_fvm = reduce(hcat, sol.u)
u_exact = reduce(hcat, u_exact)
errs = reduce(hcat, [100abs.(u - û) / maximum(abs.(u)) for (u, û) in zip(eachcol(u_exact), eachcol(u_fvm))])
@test all(<(0.1), mean.(eachcol(errs)))
@test all(<(0.1), median.(eachcol(errs)))
@test mean(errs) < 0.05
@test median(errs) < 0.05

## Step 8: Visualise the comparison 
fig = Figure(fontsize=42, resolution=(3469.8997f0, 1466.396f0))
for i in 1:5 
    ax1 = Axis(fig[1, i], width=600, height=600, title=L"(%$(join('a':'z')[i])):$ $ Exact solution, $t = %$(sol.t[i])$", titlealign=:left)
    ax2 = Axis(fig[2, i], width=600, height=600, title=L"(%$(join('a':'z')[5+i])):$ $ Numerical solution, $t = %$(sol.t[i])$", titlealign=:left)
    tricontourf!(ax1, tri, u_exact[:, i], levels = 1:0.01:1.4, colormap=:matter)
    tricontourf!(ax2, tri, u_fvm[:, i], levels = 1:0.01:1.4, colormap=:matter)
end
fig
@test_reference "../docs/src/figures/reaction_diffusion_equation_test_error.png" fig
