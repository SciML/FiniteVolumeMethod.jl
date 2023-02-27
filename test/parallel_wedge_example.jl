using ..FiniteVolumeMethod 
include("test_setup.jl")
using Test
using CairoMakie 
using OrdinaryDiffEq 
using LinearSolve
using StatsBase
using PreallocationTools

n = 50
α = π / 4
r₁ = LinRange(0, 1, n)
θ₁ = LinRange(0, 0, n)
x₁ = @. r₁ * cos(θ₁)
y₁ = @. r₁ * sin(θ₁)
r₂ = LinRange(1, 1, n)
θ₂ = LinRange(0, α, n)
x₂ = @. r₂ * cos(θ₂)
y₂ = @. r₂ * sin(θ₂)
r₃ = LinRange(1, 0, n)
θ₃ = LinRange(α, α, n)
x₃ = @. r₃ * cos(θ₃)
y₃ = @. r₃ * sin(θ₃)
x = [x₁, x₂, x₃]
y = [y₁, y₂, y₃]
r = 0.01
tri = generate_mesh(x, y, r; gmsh_path=GMSH_PATH)
points = get_points(tri)
mesh = FVMGeometry(tri)
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