using ..FiniteVolumeMethod
include("test_setup.jl")
using Test
using CairoMakie
using OrdinaryDiffEq
using ReferenceTests
using LinearSolve
using StatsBase
using StableRNGs

## Step 1: Generate the mesh 
rng = StableRNG(19191919)
a, b, c, d = 0.0, 2.0, 0.0, 2.0
p1 = (a, c)
p2 = (b, c)
p3 = (b, d)
p4 = (a, d)
points = [p1, p2, p3, p4]
boundary_nodes = [1, 2, 3, 4, 1]
tri = triangulate(points; boundary_nodes, rng)
A = get_total_area(tri)
refine!(tri; max_area=1e-4A, rng)
mesh = FVMGeometry(tri)

## Step 2: Define the boundary conditions 
bc = ((x, y, t, u::T, p) where {T}) -> zero(T)
type = :Dirichlet # or :D or :dirichlet or "D" or "Dirichlet"
BCs = BoundaryConditions(mesh, bc, type)

## Step 3: Define the actual PDE 
f = (x, y) -> y ≤ 1.0 ? 50.0 : 0.0 # initial condition 
D = (x, y, t, u, p) -> 1 / 9 # You could also define flux = (q, x, y, t, α, β, γ, p) -> (q[1] = -α/9; q[2] = -β/9)
R = ((x, y, t, u::T, p) where {T}) -> zero(T)
points = get_points(tri)
u₀ = @views f.(first.(points), last.(points))
iip_flux = true
final_time = 0.5
prob = FVMProblem(mesh, BCs; iip_flux,
    diffusion_function=D, reaction_function=R,
    initial_condition=u₀, final_time)

## Step 4: Solve
alg = TRBDF2(linsolve=KLUFactorization(), autodiff=true)
sol = solve(prob, alg; specialization=SciMLBase.FullSpecialize, saveat=0.05)

## Step 5: Visualisation 
T_mat = [T[j] for T in each_triangle(tri), j in 1:3]
fig = Figure(resolution=(2068.72f0, 686.64f0), fontsize=38)
ax = Axis(fig[1, 1], width=600, height=600)
xlims!(ax, a, b)
ylims!(ax, c, d)
mesh!(ax, points, T_mat, color=sol.u[1], colorrange=(0, 50), colormap=:matter)
ax = Axis(fig[1, 2], width=600, height=600)
xlims!(ax, a, b)
ylims!(ax, c, d)
mesh!(ax, points, T_mat, color=sol.u[6], colorrange=(0, 50), colormap=:matter)
ax = Axis(fig[1, 3], width=600, height=600)
xlims!(ax, a, b)
ylims!(ax, c, d)
mesh!(ax, points, T_mat, color=sol.u[11], colorrange=(0, 50), colormap=:matter)
@test_reference "../docs/src/figures/heat_equation_test.png" fig

## Step 6: Define the exact solution for comparison later 
function diffusion_equation_on_a_square_plate_exact_solution(x, y, t, N, M)
    u_exact = zeros(length(x))
    for j in eachindex(x)
        if t == 0.0
            if y[j] ≤ 1.0
                u_exact[j] = 50.0
            else
                u_exact[j] = 0.0
            end
        else
            u_exact[j] = 0.0
            for m = 1:M
                for n = 1:N
                    u_exact[j] += 200 / π^2 * (1 + (-1)^(m + 1)) * (1 - cos(n * π / 2)) / (m * n) * sin(m * π * x[j] / 2) * sin(n * π * y[j] / 2) * exp(-π^2 / 36 * (m^2 + n^2) * t)
                end
            end
        end
    end
    return u_exact
end

## Step 7: Compare the results
sol = solve(prob, alg; saveat=0.1)
all_errs = [Float64[] for _ in eachindex(sol)]
u_exact = [diffusion_equation_on_a_square_plate_exact_solution(first.(points), last.(points), τ, 200, 200) for τ in sol.t]
u_fvm = reduce(hcat, sol.u)
u_exact = reduce(hcat, u_exact)
errs = reduce(hcat, [100abs.(u - û) / maximum(abs.(u)) for (u, û) in zip(eachcol(u_exact), eachcol(u_fvm))])
@test all(<(0.5), mean.(eachcol(errs)))
@test all(<(0.25), median.(eachcol(errs)))
@test mean(errs) < 0.3
@test median(errs) < 0.1

## Step 8: Visualise the comparison 
# Yes there's a better way to do this
fig = Figure(fontsize=42, resolution=(3469.8997f0, 1466.396f0))
ax = Axis(fig[1, 1], width=600, height=600, title=L"(a):$ $ Exact solution, $t = %$(sol.t[1])$", titlealign=:left)
pt_mat = points
mesh!(ax, pt_mat, T_mat, color=u_exact[:, 1], colorrange=(0, 0.5), colormap=:matter)
ax = Axis(fig[1, 2], width=600, height=600, title=L"(b):$ $ Exact solution, $t = %$(sol.t[2])$", titlealign=:left)
mesh!(ax, pt_mat, T_mat, color=u_exact[:, 2], colorrange=(0, 0.5), colormap=:matter)
ax = Axis(fig[1, 3], width=600, height=600, title=L"(c):$ $ Exact solution, $t = %$(sol.t[3])$", titlealign=:left)
mesh!(ax, pt_mat, T_mat, color=u_exact[:, 3], colorrange=(0, 0.5), colormap=:matter)
ax = Axis(fig[1, 4], width=600, height=600, title=L"(d):$ $ Exact solution, $t = %$(sol.t[4])$", titlealign=:left)
mesh!(ax, pt_mat, T_mat, color=u_exact[:, 4], colorrange=(0, 0.5), colormap=:matter)
ax = Axis(fig[1, 5], width=600, height=600, title=L"(e):$ $ Exact solution, $t = %$(sol.t[5])$", titlealign=:left)
mesh!(ax, pt_mat, T_mat, color=u_exact[:, 5], colorrange=(0, 0.5), colormap=:matter)
ax = Axis(fig[2, 1], width=600, height=600, title=L"(f):$ $ Numerical solution, $t = %$(sol.t[1])$", titlealign=:left)
mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 1], colorrange=(0, 0.5), colormap=:matter)
ax = Axis(fig[2, 2], width=600, height=600, title=L"(g):$ $ Numerical solution, $t = %$(sol.t[2])$", titlealign=:left)
mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 2], colorrange=(0, 0.5), colormap=:matter)
ax = Axis(fig[2, 3], width=600, height=600, title=L"(h):$ $ Numerical solution, $t = %$(sol.t[3])$", titlealign=:left)
mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 3], colorrange=(0, 0.5), colormap=:matter)
ax = Axis(fig[2, 4], width=600, height=600, title=L"(i):$ $ Numerical solution, $t = %$(sol.t[4])$", titlealign=:left)
mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 4], colorrange=(0, 0.5), colormap=:matter)
ax = Axis(fig[2, 5], width=600, height=600, title=L"(j):$ $ Numerical solution, $t = %$(sol.t[5])$", titlealign=:left)
mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 5], colorrange=(0, 0.5), colormap=:matter)
@test_reference "../docs/src/figures/heat_equation_test_error.png" fig
