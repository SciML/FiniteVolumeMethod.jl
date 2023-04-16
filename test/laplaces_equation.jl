using ..FiniteVolumeMethod
include("test_setup.jl")
using CairoMakie
using Test
using OrdinaryDiffEq
using LinearSolve
using StatsBase
using SteadyStateDiffEq
using ReferenceTests
using StableRNGs
using Krylov

## Define the problem
a, b = 0.0, 1.0π
c, d = 0.0, 1.0π
p1 = (a, c)
p2 = (b, c)
p3 = (b, d)
p4 = (a, d)
points = [p1, p2, p3, p4]
rng = StableRNG(19191919)
tri = triangulate(points; boundary_nodes=[[1, 2], [2, 3], [3, 4], [4, 1]], rng)
refine!(tri; max_area=1e-4get_total_area(tri), rng)
mesh = FVMGeometry(tri)
bc1 = (x, y, t, u, p) -> sinh(x)
bc2 = (x, y, t, u, p) -> sinh(π) * cos(y)
bc3 = (x, y, t, u, p) -> -sinh(x)
bc4 = (x, y, t, u, p) -> zero(u)
types = [:D, :D, :D, :D]
BCs = BoundaryConditions(mesh, [bc1, bc2, bc3, bc4], types)
diffusion_function = (x, y, t, u, p) -> one(u)
final_time = Inf
initial_guess = zeros(num_points(tri))
for (i, f) in pairs((bc1, bc2, bc3, bc4))
    bn = get_boundary_nodes(tri, i)
    for j in bn
        p = get_point(tri, j)
        x, y = getxy(p)
        initial_guess[j] = f(x, y, Inf, 0.0, nothing)
    end
end
prob = FVMProblem(mesh, BCs;
    diffusion_function,
    final_time,
    initial_condition=initial_guess,
    steady=true)

## Solve with a steady state algorithm
alg = DynamicSS(TRBDF2(linsolve=KrylovJL_GMRES()))
sol = solve(prob, alg, parallel=true)

## Plot 
fig = Figure(fontsize=38)
ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"y")
pt_mat = get_points(tri)
T_mat = [T[j] for T in each_triangle(tri), j in 1:3]
msh = mesh!(ax, pt_mat, T_mat, color=sol, colorrange=(-15, 15))
Colorbar(fig[1, 2], msh)
@test_reference "../docs/src/figures/laplace_equation.png" fig

## Test 
exact = [sinh(x) * cos(y) for (x, y) in each_point(tri)]
error_fnc = (fvm_sol, exact_sol) -> 100abs(fvm_sol - exact_sol) / maximum(exact)
@test median(error_fnc.(exact, sol)) < 0.2
