using ..FiniteVolumeMethod 
include("test_setup.jl")
using Test
using CairoMakie 
using OrdinaryDiffEq 
using LinearSolve
using StatsBase
using ElasticArrays
using Bessels
using StableRNGs

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
bc = (x, y, t, u, p) -> u
types = :dudt
BCs = BoundaryConditions(mesh, bc, types)
f = (x, y) -> sqrt(besseli(0.0, sqrt(2) * sqrt(x^2 + y^2)))
D = (x, y, t, u, p) -> u
R = (x, y, t, u, p) -> u * (1 - u)
u₀ = [f(points[:, i]...) for i in axes(points, 2)]
final_time = 0.5
prob = FVMProblem(mesh, BCs; iip_flux=false, diffusion_function=D, reaction_function=R, initial_condition=u₀, final_time)
alg = TRBDF2(linsolve=KLUFactorization(; reuse_symbolic=false))
sol_par = solve(prob, alg; parallel=true, saveat=0.05)
sol_ser = solve(prob, alg; parallel=false, saveat=0.05)
@test sol_par.u ≈ sol_ser.u rtol=1e-7

prob = FVMProblem(mesh, BCs; iip_flux=true, diffusion_function=D, reaction_function=R, initial_condition=u₀, final_time)
alg = TRBDF2(linsolve=KLUFactorization(; reuse_symbolic=false))
sol_par = solve(prob, alg; parallel=true, saveat=0.05)
sol_ser = solve(prob, alg; parallel=false, saveat=0.05)
@test sol_par.u ≈ sol_ser.u rtol=1e-7