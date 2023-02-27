using ..FiniteVolumeMethod 
include("test_setup.jl")
using Test
using CairoMakie 
using OrdinaryDiffEq 
using LinearSolve
using StatsBase
using Bessels

n = 50
r = LinRange(1, 1, 1000)
θ = LinRange(0, 2π, 1000)
x = @. r * cos(θ)
y = @. r * sin(θ)
r = 0.02
tri = generate_mesh(x, y, r; gmsh_path=GMSH_PATH)
points = get_points(tri)
mesh = FVMGeometry(tri)
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