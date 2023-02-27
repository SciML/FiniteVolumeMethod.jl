using ..FiniteVolumeMethod
include("test_setup.jl")
using Test
using CairoMakie
using OrdinaryDiffEq
using LinearSolve
using StatsBase

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
n = 5
x₁ = LinRange(-L, L, n)
x₂ = LinRange(L, L, n)
x₃ = LinRange(L, -L, n)
x₄ = LinRange(-L, -L, n)
y₁ = LinRange(-L, -L, n)
y₂ = LinRange(-L, L, n)
y₃ = LinRange(L, L, n)
y₄ = LinRange(L, -L, n)
x = reduce(vcat, [x₁, x₂, x₃, x₄])
y = reduce(vcat, [y₁, y₂, y₃, y₄])
xy = [(x, y) for (x, y) in zip(x, y)]
unique!(xy)
x = getx.(xy)
y = gety.(xy)
r = 0.07
tri = generate_mesh(x, y, r; gmsh_path=GMSH_PATH)
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
u₀ = [f(points[:, i]...) for i in axes(points, 2)]
prob = FVMProblem(mesh, BCs; diffusion_function=diff_fnc,
    diffusion_parameters=diff_parameters,
    reaction_function=reac_fnc, reaction_parameters=react_parameter,
    initial_condition=u₀, final_time)

## Step 5: Solve
alg = TRBDF2(linsolve=KLUFactorization())
sol = solve(prob, alg; saveat=2.5)

## Step 6: Visualisation 
pt_mat = Matrix(points')
T_mat = [T[j] for T in each_triangle(tri), j in 1:3]
fig = Figure(resolution=(2131.8438f0, 684.27f0), fontsize=38)
ax = Axis(fig[1, 1], width=600, height=600)
mesh!(ax, pt_mat, T_mat, color=sol.u[1], colorrange=(0.0, 0.5), colormap=:matter)
ax = Axis(fig[1, 2], width=600, height=600)
mesh!(ax, pt_mat, T_mat, color=sol.u[3], colorrange=(0.0, 0.5), colormap=:matter)
ax = Axis(fig[1, 3], width=600, height=600)
mesh!(ax, pt_mat, T_mat, color=sol.u[5], colorrange=(0.0, 0.5), colormap=:matter)
SAVE_FIGURE && save("figures/porous_medium_linear_source_test.png", fig)

## Step 7: Compare the results
all_errs = [Float64[] for _ in eachindex(sol)]
u_exact = [porous_medium_linear_source_exact_solution(points[1, :], points[2, :], τ, m, M, D, λ) for τ in sol.t]
u_fvm = reduce(hcat, sol.u)
u_exact = reduce(hcat, u_exact)
errs = reduce(hcat, [100abs.(u - û) / maximum(abs.(u)) for (u, û) in zip(eachcol(u_exact[:, 2:end]), eachcol(u_fvm[:, 2:end]))])
@test all(<(0.26), mean.(eachcol(errs)))
@test all(<(0.17), median.(eachcol(errs)))
@test mean(errs) < 0.25
@test median(errs) < 0.05

## Step 8: Visualise the comparison 
fig = Figure(fontsize=42, resolution=(3586.6597f0, 1466.396f0))
ax = Axis(fig[1, 1], width=600, height=600, title=L"(a):$ $ Exact solution, $t = %$(sol.t[1])$", titlealign=:left)
mesh!(ax, pt_mat, T_mat, color=u_exact[:, 1], colorrange=(0, 0.05), colormap=:matter)
ax = Axis(fig[1, 2], width=600, height=600, title=L"(b):$ $ Exact solution, $t = %$(sol.t[2])$", titlealign=:left)
mesh!(ax, pt_mat, T_mat, color=u_exact[:, 2], colorrange=(0, 0.05), colormap=:matter)
ax = Axis(fig[1, 3], width=600, height=600, title=L"(c):$ $ Exact solution, $t = %$(sol.t[3])$", titlealign=:left)
mesh!(ax, pt_mat, T_mat, color=u_exact[:, 3], colorrange=(0, 0.05), colormap=:matter)
ax = Axis(fig[1, 4], width=600, height=600, title=L"(d):$ $ Exact solution, $t = %$(sol.t[4])$", titlealign=:left)
mesh!(ax, pt_mat, T_mat, color=u_exact[:, 4], colorrange=(0, 0.05), colormap=:matter)
ax = Axis(fig[1, 5], width=600, height=600, title=L"(e):$ $ Exact solution, $t = %$(sol.t[5])$", titlealign=:left)
mesh!(ax, pt_mat, T_mat, color=u_exact[:, 5], colorrange=(0, 0.05), colormap=:matter)
ax = Axis(fig[2, 1], width=600, height=600, title=L"(f):$ $ Numerical solution, $t = %$(sol.t[1])$", titlealign=:left)
mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 1], colorrange=(0, 0.05), colormap=:matter)
ax = Axis(fig[2, 2], width=600, height=600, title=L"(g):$ $ Numerical solution, $t = %$(sol.t[2])$", titlealign=:left)
mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 2], colorrange=(0, 0.05), colormap=:matter)
ax = Axis(fig[2, 3], width=600, height=600, title=L"(h):$ $ Numerical solution, $t = %$(sol.t[3])$", titlealign=:left)
mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 3], colorrange=(0, 0.05), colormap=:matter)
ax = Axis(fig[2, 4], width=600, height=600, title=L"(i):$ $ Numerical solution, $t = %$(sol.t[4])$", titlealign=:left)
mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 4], colorrange=(0, 0.05), colormap=:matter)
ax = Axis(fig[2, 5], width=600, height=600, title=L"(j):$ $ Numerical solution, $t = %$(sol.t[5])$", titlealign=:left)
mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 5], colorrange=(0, 0.05), colormap=:matter)
SAVE_FIGURE && save("figures/porous_medium_linear_source_test_error.png", fig)
