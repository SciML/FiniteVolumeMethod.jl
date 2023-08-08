using ..FiniteVolumeMethod
include("test_setup.jl")
using Test
using CairoMakie
using OrdinaryDiffEq
using LinearSolve
using StatsBase
using StableRNGs
using ReferenceTests

## Step 0: Define all the parameters 
m = 2
M = 0.37
D = 2.53
final_time = 12.0
ε = 0.1

## Step 1: Define the mesh 
RmM = 4m / (m - 1) * (M / (4π))^((m - 1) / m)
L = sqrt(RmM) * (D * final_time)^(1 / (2m))
x = [-L, L, L, -L, -L]
y = [-L, -L, L, L, -L]
boundary_nodes, points = convert_boundary_points_to_indices(x, y)
rng = StableRNG(123)
tri = triangulate(points; boundary_nodes, rng)
A = get_total_area(tri)
refine!(tri; max_area=1e-4A / 2, rng)
mesh = FVMGeometry(tri)
points = get_points(tri)

## Step 2: Define the boundary conditions 
bc = ((x, y, t, u::T, p) where {T}) -> zero(T)
types = :D
BCs = BoundaryConditions(mesh, bc, types)

## Step 3: Define the actual PDE  
f = (x, y) -> M * 1 / (ε^2 * π) * exp(-1 / (ε^2) * (x^2 + y^2))
diff_fnc = (x, y, t, u, p) -> p[1] * u^(p[2] - 1)
diff_parameters = (D, m)
u₀ = f.(first.(points), last.(points))
prob = FVMProblem(mesh, BCs; diffusion_function=diff_fnc,
    diffusion_parameters=diff_parameters, initial_condition=u₀, final_time)

## Step 4: Solve
alg = TRBDF2(linsolve=KLUFactorization())
sol = solve(prob, alg; saveat=3.0)

## Step 5: Visualisation 
fig = Figure(resolution=(2068.72f0, 686.64f0), fontsize=38)
for (i, j) in zip(1:3, (1, 3, 5))
    ax = Axis(fig[1, i], width=600, height=600)
    tricontourf!(ax, tri, sol.u[j], levels=0:0.005:0.05, colormap=:matter, extendhigh=:auto)
    tightlimits!(ax)
end
fig
@test_reference "../docs/src/figures/porous_medium_test.png" fig

## Step 6: Define the exact solution for comparison later 
function porous_medium_exact_solution(x, y, t, m, M, D)
    u_exact = zeros(length(x))
    RmM = 4m / (m - 1) * (M / (4π))^((m - 1) / m)
    for i in eachindex(x)
        if x[i]^2 + y[i]^2 < RmM * (D * t)^(1 / m)
            u_exact[i] = (D * t)^(-1 / m) * ((M / (4π))^((m - 1) / m) - (m - 1) / (4m) * (x[i]^2 + y[i]^2) * (D * t)^(-1 / m))^(1 / (m - 1))
        else
            u_exact[i] = 0.0
        end
    end
    return u_exact
end

## Step 7: Compare the results
all_errs = [Float64[] for _ in eachindex(sol)]
u_exact = [porous_medium_exact_solution(first.(points), last.(points), τ, m, M, D) for τ in sol.t]
u_fvm = reduce(hcat, sol.u)
u_exact = reduce(hcat, u_exact)
errs = reduce(hcat, [100abs.(u - û) / maximum(abs.(u)) for (u, û) in zip(eachcol(u_exact[:, 2:end]), eachcol(u_fvm[:, 2:end]))])
@test all(<(0.1), mean.(eachcol(errs)))
@test all(<(0.1), median.(eachcol(errs)))
@test mean(errs) < 0.06
@test median(errs) < 0.06

## Step 8: Visualise the comparison 
fig = Figure(fontsize=42, resolution=(3469.8997f0, 1466.396f0))
for i in 1:5
    ax1 = Axis(fig[1, i], width=600, height=600, title=L"(%$(join('a':'z')[i])):$ $ Exact solution, $t = %$(sol.t[i])$", titlealign=:left)
    ax2 = Axis(fig[2, i], width=600, height=600, title=L"(%$(join('a':'z')[5+i])):$ $ Numerical solution, $t = %$(sol.t[i])$", titlealign=:left)
    tricontourf!(ax1, tri, u_exact[:, i], levels=0:0.005:0.05, colormap=:matter, extendhigh=:auto)
    tricontourf!(ax2, tri, u_fvm[:, i], levels=0:005:0.05, colormap=:matter, extendhigh=:auto)
end
fig
@test_reference "../docs/src/figures/porous_medium_test_error.png" fig
