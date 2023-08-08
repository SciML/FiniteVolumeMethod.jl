using ..FiniteVolumeMethod 
include("test_setup.jl")
using Test
using CairoMakie 
using OrdinaryDiffEq 
using LinearSolve
using StatsBase
using FastGaussQuadrature 
using Bessels
using ReferenceTests
using Cubature
using StableRNGs
using ElasticArrays

## Step 1: Generate the mesh 
n = 50
α = π / 4

# The bottom edge 
x₁ = [0.0,1.0]
y₁ = [0.0,0.0]

# Arc 
r₂ = LinRange(1, 1, n)
θ₂ = LinRange(0, α, n)
x₂ = @. r₂ * cos(θ₂)
y₂ = @. r₂ * sin(θ₂)

# Upper edge 
x₃ = [cos(α), 0.0]
y₃ = [sin(α), 0.0]

# Combine and create the mesh 
x = [x₁, x₂, x₃]
y = [y₁, y₂, y₃]
boundary_nodes, points = convert_boundary_points_to_indices(x, y; existing_points = ElasticMatrix{Float64}(undef, 2, 0))
rng = StableRNG(191919198888)
tri = triangulate(points; boundary_nodes, rng)
A = get_total_area(tri)
refine!(tri; max_area = 1e-4A, rng)
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
u₀ = f.(points[1, :], points[2, :]) |> collect
final_time = 0.1 # Do not need iip_flux = true or R(x, y, t, u, p) = 0, these are defaults 
prob = FVMProblem(mesh, BCs; diffusion_function=D, initial_condition=u₀, final_time)

flux = (q, x, y, t, α, β, γ, p) -> (q[1] = -α; q[2] = -β; nothing)
prob2 = FVMProblem(mesh, BCs; diffusion_function=D, initial_condition=u₀, final_time)

## Step 4: Solve
alg = Rosenbrock23(linsolve=UMFPACKFactorization())
sol = solve(prob, alg; saveat=0.025)
sol2 = solve(prob2, alg; saveat=0.025)

## Step 5: Visualisation 
fig = Figure(resolution=(2068.72f0, 686.64f0), fontsize=38)
for (i, j) in zip(1:3, (1, 3, 5))
    ax = Axis(fig[1, i], width=600, height=600)
    tricontourf!(ax, tri, sol.u[j], levels=0:0.01:1, colormap=:matter)
    tightlimits!(ax)
end
fig
@test_reference "../docs/src/figures/diffusion_equation_wedge_test.png" fig

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
fig = Figure(fontsize=42, resolution=(3469.8997f0, 1466.396f0))
for i in 1:5 
    ax1 = Axis(fig[1, i], width=600, height=600, title=L"(%$(join('a':'z')[i])):$ $ Exact solution, $t = %$(sol.t[i])$", titlealign=:left)
    ax2 = Axis(fig[2, i], width=600, height=600, title=L"(%$(join('a':'z')[5+i])):$ $ Numerical solution, $t = %$(sol.t[i])$", titlealign=:left)
    tricontourf!(ax1, tri, u_exact[:, i], levels = 0:0.01:1, colormap=:matter)
    tricontourf!(ax2, tri, u_fvm[:, i], levels = 0:0.01:1, colormap=:matter)
end
fig
@test_reference "../docs/src/figures/heat_equation_wedge_test_error.png" fig