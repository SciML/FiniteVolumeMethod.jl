using ..FiniteVolumeMethod 
include("test_setup.jl")
using Test
using CairoMakie 
using OrdinaryDiffEq 
using LinearSolve
using StatsBase
using FastGaussQuadrature 
using Bessels
using Cubature

## Step 1: Generate the mesh 
n = 50
α = π / 4

# The bottom edge 
r₁ = LinRange(0, 1, n)
θ₁ = LinRange(0, 0, n)
x₁ = @. r₁ * cos(θ₁)
y₁ = @. r₁ * sin(θ₁)

# Arc 
r₂ = LinRange(1, 1, n)
θ₂ = LinRange(0, α, n)
x₂ = @. r₂ * cos(θ₂)
y₂ = @. r₂ * sin(θ₂)

# Upper edge 
r₃ = LinRange(1, 0, n)
θ₃ = LinRange(α, α, n)
x₃ = @. r₃ * cos(θ₃)
y₃ = @. r₃ * sin(θ₃)

# Combine and create the mesh 
x = [x₁, x₂, x₃]
y = [y₁, y₂, y₃]
r = 0.01
tri = generate_mesh(x, y, r; gmsh_path=GMSH_PATH)
mesh = FVMGeometry(tri)

## Step 2: Define the boundary conditions 
lower_bc = ((x, y, t, u::T, p) where {T}) -> zero(T)
arc_bc = ((x, y, t, u::T, p) where {T}) -> zero(T)
upper_bc = ((x, y, t, u::T, p) where {T}) -> zero(T)
types = (:N, :D, :N)
boundary_functions = (lower_bc, arc_bc, upper_bc)
BCs = BoundaryConditions(mesh, boundary_functions, types)

## Step 3: Define the actual PDE  
f = (x, y) -> 1 - sqrt(x^2 + y^2)
D = ((x, y, t, u::T, p) where {T}) -> one(T)
points = get_points(tri)
u₀ = f.(points[1, :], points[2, :])
final_time = 0.1 # Do not need iip_flux = true or R(x, y, t, u, p) = 0, these are defaults 
prob = FVMProblem(mesh, BCs; diffusion_function=D, initial_condition=u₀, final_time)

flux = (q, x, y, t, α, β, γ, p) -> (q[1] = -α; q[2] = -β; nothing)
prob2 = FVMProblem(mesh, BCs; diffusion_function=D, initial_condition=u₀, final_time)

## Step 4: Solve
alg = Rosenbrock23(linsolve=UMFPACKFactorization())
sol = solve(prob, alg; saveat=0.025)
sol2 = solve(prob2, alg; saveat=0.025)

## Step 5: Visualisation 
pt_mat = Matrix(points')
T_mat = [T[j] for T in each_triangle(tri), j in 1:3]
fig = Figure(resolution=(2131.8438f0, 684.27f0), fontsize=38)
ax = Axis(fig[1, 1], width=600, height=600)
mesh!(ax, pt_mat, T_mat, color=sol.u[1], colorrange=(0, 0.5), colormap=:matter)
ax = Axis(fig[1, 2], width=600, height=600)
mesh!(ax, pt_mat, T_mat, color=sol.u[3], colorrange=(0, 0.5), colormap=:matter)
ax = Axis(fig[1, 3], width=600, height=600)
mesh!(ax, pt_mat, T_mat, color=sol.u[5], colorrange=(0, 0.5), colormap=:matter)
SAVE_FIGURE && save("figures/diffusion_equation_wedge_test.png", fig)

## Step 6: Define the exact solution for comparison later 
function diffusion_equation_on_a_wedge_exact_solution(x, y, t, α, N, M)
    f = (r, θ) -> 1.0 - r
    ## Compute the ζ: ζ[m, n+1] is the mth zero of the (nπ/α)th order Bessel function of the first kind 
    ζ = zeros(M, N + 2)
    for n in 0:(N+1)
        order = n * π / α
        @views ζ[:, n+1] .= approx_besselroots(order, M)
    end
    A = zeros(M, N + 1) # A[m, n+1] is the coefficient Aₙₘ
    for n in 0:N
        order = n * π / α
        for m in 1:M
            integrand = rθ -> f(rθ[2], rθ[1]) * besselj(order, ζ[m, n+1] * rθ[2]) * cos(order * rθ[1]) * rθ[2]
            A[m, n+1] = 4.0 / (α * besselj(order + 1, ζ[m, n+1])^2) * hcubature(integrand, [0.0, 0.0], [α, 1.0]; abstol=1e-8)[1]
        end
    end
    r = @. sqrt(x^2 + y^2)
    θ = @. atan(y, x)
    u_exact = zeros(length(x))
    for i in 1:length(x)
        for m = 1:M
            u_exact[i] = u_exact[i] + 0.5 * A[m, 1] * exp(-ζ[m, 1]^2 * t) * besselj(0.0, ζ[m, 1] * r[i])
        end
        for n = 1:N
            order = n * π / α
            for m = 1:M
                u_exact[i] = u_exact[i] + A[m, n+1] * exp(-ζ[m, n+1]^2 * t) * besselj(order, ζ[m, n+1] * r[i]) * cos(order * θ[i])
            end
        end
    end
    return u_exact
end

## Step 7: Compare the results
all_errs = [Float64[] for _ in eachindex(sol)]
u_exact = [diffusion_equation_on_a_wedge_exact_solution(points[1, :], points[2, :], τ, α, 22, 24) for τ in sol.t]
u_fvm = reduce(hcat, sol.u)
u_exact = reduce(hcat, u_exact)
_u_exact = deepcopy(u_exact)
errs = reduce(hcat, [100abs.(u - û) / maximum(abs.(u)) for (u, û) in zip(eachcol(u_exact), eachcol(u_fvm))])
@test all(<(0.3), mean.(eachcol(errs)))
@test all(<(0.15), median.(eachcol(errs)))
@test mean(errs) < 0.15
@test median(errs) < 0.1

all_errs2 = [Float64[] for _ in eachindex(sol2)]
u_exact2 = [diffusion_equation_on_a_wedge_exact_solution(points[1, :], points[2, :], τ, α, 22, 24) for τ in sol2.t]
u_fvm2 = reduce(hcat, sol2.u)
u_exact2 = reduce(hcat, u_exact2)
errs2 = reduce(hcat, [100abs.(u - û) / maximum(abs.(u)) for (u, û) in zip(eachcol(u_exact2), eachcol(u_fvm2))])
@test errs == errs2
@test u_fvm2 == u_fvm

## Step 8: Visualise the comparison 
fig = Figure(fontsize=42, resolution=(3586.6597f0, 1466.396f0))
ax = Axis(fig[1, 1], width=600, height=600, title=L"(a):$ $ Exact solution, $t = %$(sol.t[1])$", titlealign=:left)
mesh!(ax, pt_mat, T_mat, color=_u_exact[:, 1], colorrange=(0, 0.5), colormap=:matter)
ax = Axis(fig[1, 2], width=600, height=600, title=L"(b):$ $ Exact solution, $t = %$(sol.t[2])$", titlealign=:left)
mesh!(ax, pt_mat, T_mat, color=_u_exact[:, 2], colorrange=(0, 0.5), colormap=:matter)
ax = Axis(fig[1, 3], width=600, height=600, title=L"(c):$ $ Exact solution, $t = %$(sol.t[3])$", titlealign=:left)
mesh!(ax, pt_mat, T_mat, color=_u_exact[:, 3], colorrange=(0, 0.5), colormap=:matter)
ax = Axis(fig[1, 4], width=600, height=600, title=L"(d):$ $ Exact solution, $t = %$(sol.t[4])$", titlealign=:left)
mesh!(ax, pt_mat, T_mat, color=_u_exact[:, 4], colorrange=(0, 0.5), colormap=:matter)
ax = Axis(fig[1, 5], width=600, height=600, title=L"(e):$ $ Exact solution, $t = %$(sol.t[5])$", titlealign=:left)
mesh!(ax, pt_mat, T_mat, color=_u_exact[:, 5], colorrange=(0, 0.5), colormap=:matter)
ax = Axis(fig[2, 1], width=600, height=600, title=L"(f):$ $ Numerical solution, $t = %$(sol.t[1])$", titlealign=:left)
mesh!(ax, pt_mat, T_mat, color=sol.u[1], colorrange=(0, 0.5), colormap=:matter)
ax = Axis(fig[2, 2], width=600, height=600, title=L"(g):$ $ Numerical solution, $t = %$(sol.t[2])$", titlealign=:left)
mesh!(ax, pt_mat, T_mat, color=sol.u[2], colorrange=(0, 0.5), colormap=:matter)
ax = Axis(fig[2, 3], width=600, height=600, title=L"(h):$ $ Numerical solution, $t = %$(sol.t[3])$", titlealign=:left)
mesh!(ax, pt_mat, T_mat, color=sol.u[3], colorrange=(0, 0.5), colormap=:matter)
ax = Axis(fig[2, 4], width=600, height=600, title=L"(i):$ $ Numerical solution, $t = %$(sol.t[4])$", titlealign=:left)
mesh!(ax, pt_mat, T_mat, color=sol.u[4], colorrange=(0, 0.5), colormap=:matter)
ax = Axis(fig[2, 5], width=600, height=600, title=L"(j):$ $ Numerical solution, $t = %$(sol.t[5])$", titlealign=:left)
mesh!(ax, pt_mat, T_mat, color=sol.u[5], colorrange=(0, 0.5), colormap=:matter)
SAVE_FIGURE && save("figures/heat_equation_wedge_test_error.png", fig)
