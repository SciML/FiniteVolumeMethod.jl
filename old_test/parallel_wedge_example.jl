using ..FiniteVolumeMethod 
include("test_setup.jl")
using Test
using CairoMakie 
using OrdinaryDiffEq 
using LinearSolve
using StatsBase
using PreallocationTools
using StableRNGs
using ElasticArrays

n = 50
α = π / 4
x₁ = [0.0,1.0]
y₁ = [0.0,0.0]
r₂ = LinRange(1, 1, n)
θ₂ = LinRange(0, α, n)
x₂ = @. r₂ * cos(θ₂)
y₂ = @. r₂ * sin(θ₂)
x₃ = [cos(α), 0.0]
y₃ = [sin(α), 0.0]
x = [x₁, x₂, x₃]
y = [y₁, y₂, y₃]
boundary_nodes, points = convert_boundary_points_to_indices(x, y; existing_points = ElasticMatrix{Float64}(undef, 2, 0))
rng = StableRNG(191919198888)
tri = triangulate(points; boundary_nodes, rng)
A = get_total_area(tri)
refine!(tri; max_area = 1e-4A, rng)
mesh = FVMGeometry(tri)
points = get_points(tri)
lower_bc = ((x, y, t, u::T, p) where {T}) -> zero(T)
arc_bc = ((x, y, t, u::T, p) where {T}) -> zero(T)
upper_bc = ((x, y, t, u::T, p) where {T}) -> zero(T)
types = (:N, :D, :N)
boundary_functions = (lower_bc, arc_bc, upper_bc)
BCs = BoundaryConditions(mesh, boundary_functions, types)
f = (x, y) -> 1 - sqrt(x^2 + y^2)
D = ((x, y, t, u::T, p) where {T}) -> one(T)
u₀ = f.(points[1, :], points[2, :])
final_time = 20.0
prob = FVMProblem(mesh, BCs; iip_flux=true, diffusion_function=D, initial_condition=u₀, final_time)
sol_par = solve(prob, TRBDF2(linsolve=KLUFactorization(), autodiff=true); parallel=true, saveat=0.05)
sol_ser = solve(prob, TRBDF2(linsolve=KLUFactorization(), autodiff=true); parallel=false, saveat=0.05)
@test sol_par.u ≈ sol_ser.u