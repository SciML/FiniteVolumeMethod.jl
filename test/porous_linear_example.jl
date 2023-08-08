using ..FiniteVolumeMethod
include("test_setup.jl")
using Test
using CairoMakie
using OrdinaryDiffEq
using LinearSolve
using StatsBase
using ReferenceTests
using StableRNGs

## Step 0: Define all the parameters 
m = 3.4
M = 2.3
D = 0.581
λ = 0.2
final_time = 10.0
ε = 0.1

## Step 1: Define the mesh 
RmM = 4m / (m - 1) * (M / (4π))^((m - 1) / m)
L = sqrt(RmM) * (D / (λ * (m - 1)) * (exp(λ * (m - 1) * final_time) - 1))^(1 / (2m))
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

## Step 3: Define the exact solution for comparison later 
function porous_medium_exact_solution(x, y, t, m, M)
    u_exact = zeros(length(x))
    RmM = 4m / (m - 1) * (M / (4π))^((m - 1) / m)
    for i in eachindex(x)
        if x[i]^2 + y[i]^2 < RmM * t^(1 / m)
            u_exact[i] = t^(-1 / m) * ((M / (4π))^((m - 1) / m) - (m - 1) / (4m) * (x[i]^2 + y[i]^2) * t^(-1 / m))^(1 / (m - 1))
        else
            u_exact[i] = 0.0
        end
    end
    return u_exact
end
function porous_medium_linear_source_exact_solution(x, y, t, m, M, D, λ)
    return exp(λ * t) * porous_medium_exact_solution(x, y, D / (λ * (m - 1)) * (exp(λ * (m - 1) * t) - 1), m, M)
end

## Step 4: Define the actual PDE  
f = (x, y) -> M * 1 / (ε^2 * π) * exp(-1 / (ε^2) * (x^2 + y^2))
diff_fnc = (x, y, t, u, p) -> p[1] * abs(u)^(p[2] - 1)
reac_fnc = (x, y, t, u, p) -> p[1] * u
diff_parameters = (D, m)
react_parameter = λ
u₀ = f.(first.(points), last.(points))
prob = FVMProblem(mesh, BCs; diffusion_function=diff_fnc,
    diffusion_parameters=diff_parameters,
    reaction_function=reac_fnc, reaction_parameters=react_parameter,
    initial_condition=u₀, final_time)

## Step 5: Solve
alg = TRBDF2(linsolve=KLUFactorization())
sol = solve(prob, alg; saveat=2.5)

## Step 6: Visualisation 
fig = Figure(resolution=(2132.72f0, 686.64f0), fontsize=38)
for (i, j) in zip(1:3, (1, 3, 5))
    ax = Axis(fig[1, i], width=600, height=600)
    tricontourf!(ax, tri, sol.u[j], levels=0:0.05:1, colormap=:matter, extendhigh=:auto)
    tightlimits!(ax)
end
fig
@test_reference "../docs/src/figures/porous_medium_linear_source_test.png" fig

## Step 7: Compare the results
all_errs = [Float64[] for _ in eachindex(sol)]
u_exact = [porous_medium_linear_source_exact_solution(first.(points), last.(points), τ, m, M, D, λ) for τ in sol.t]
u_fvm = reduce(hcat, sol.u)
u_exact = reduce(hcat, u_exact)
errs = reduce(hcat, [100abs.(u - û) / maximum(abs.(u)) for (u, û) in zip(eachcol(u_exact[:, 2:end]), eachcol(u_fvm[:, 2:end]))])
@test all(<(0.26), mean.(eachcol(errs)))
@test all(<(0.17), median.(eachcol(errs)))
@test mean(errs) < 0.25
@test median(errs) < 0.05

## Step 8: Visualise the comparison 
fig = Figure(fontsize=42, resolution=(3469.8997f0, 1466.396f0))
for i in 1:5
    ax1 = Axis(fig[1, i], width=600, height=600, title=L"(%$(join('a':'z')[i])):$ $ Exact solution, $t = %$(sol.t[i])$", titlealign=:left)
    ax2 = Axis(fig[2, i], width=600, height=600, title=L"(%$(join('a':'z')[5+i])):$ $ Numerical solution, $t = %$(sol.t[i])$", titlealign=:left)
    tricontourf!(ax1, tri, u_exact[:, i], levels=0:0.05:1, colormap=:matter, extendhigh=:auto)
    tricontourf!(ax2, tri, u_fvm[:, i], levels=0:0.05:1, colormap=:matter, extendhigh=:auto)
end
fig
@test_reference "../docs/src/figures/porous_medium_linear_source_test_error.png" fig
