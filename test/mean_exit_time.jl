using ..FiniteVolumeMethod
include("test_setup.jl")
using CairoMakie
using LinearSolve
using OrdinaryDiffEq
using LinearAlgebra
using Test
using NonlinearSolve
using Krylov
using StatsBase
using SteadyStateDiffEq

## Circle problem 
using ..FiniteVolumeMethod
using DelaunayTriangulation
using CairoMakie
using OrdinaryDiffEq
using SteadyStateDiffEq
using Test
using LinearAlgebra
using StatsBase
include("test_setup.jl")
R = 1.0
θ = LinRange(0, 2π, 100)
x = R .* cos.(θ)
y = R .* sin.(θ)
tri = generate_mesh(x, y, 0.2; gmsh_path=GMSH_PATH)
mesh = FVMGeometry(tri)
bc = (x, y, t, u, p) -> zero(u)
type = :D
BCs = BoundaryConditions(mesh, bc, type)
D = 2.5e-5
initial_guess = ones(num_points(tri))
initial_guess[get_boundary_nodes(tri)] .= 0.0
diffusion_function = (x, y, t, u, D) -> D
diffusion_parameters = D
reaction = (x, y, t, u, p) -> one(u)
final_time = Inf # doesn't matter in this case, but we still need to provide it
prob = FVMProblem(mesh, BCs;
    diffusion_function,
    diffusion_parameters,
    reaction_function=reaction,
    final_time=final_time,
    initial_condition=initial_guess,
    steady=true)

# Solve 
alg = DynamicSS(Rosenbrock23())
sol = solve(prob, alg)

# Visualise 
fig = Figure(fontsize=38)
ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"y")
pt_mat = Matrix(get_points(tri)')
T_mat = [T[j] for T in each_triangle(tri), j in 1:3]
msh = mesh!(ax, pt_mat, T_mat, color=sol, colorrange=(0, 10000))
Colorbar(fig[1, 2], msh)
SAVE_FIGURE && save("figures/circle_mean_exit_time.png", fig)

# Test 
exact = [(R^2 - norm(p)^2) / (4D) for p in each_point(tri)]
error_fnc = (fvm_sol, exact_sol) -> 100abs(fvm_sol - exact_sol) / maximum(exact)
@test median(error_fnc.(exact, sol)) < 0.2

## Perturbed circle 
using ..FiniteVolumeMethod
using DelaunayTriangulation
using CairoMakie
using OrdinaryDiffEq
using SteadyStateDiffEq
using LinearSolve
using FastLapackInterface
include("test_setup.jl")
R = 1.0
θ = LinRange(0, 2π, 100)
ε = 1 / 20
g = θ -> sin(3θ) + cos(5θ) - sin(θ)
R_bnd = 1 .+ ε .* g.(θ)
x = R_bnd .* cos.(θ)
y = R_bnd .* sin.(θ)
tri = generate_mesh(x, y, 0.2; gmsh_path=GMSH_PATH)
mesh = FVMGeometry(tri)
bc = (x, y, t, u, p) -> zero(u)
type = :D
BCs = BoundaryConditions(mesh, bc, type)
D = 2.5e-5
initial_guess = [(R^2 - norm(p)^2) / (4D) for p in each_point(tri)]
initial_guess[get_boundary_nodes(tri)] .= 0.0
diffusion_function = (x, y, t, u, D) -> D
diffusion_parameters = D
reaction = (x, y, t, u, p) -> one(u)
final_time = Inf # doesn't matter in this case, but we still need to provide it
prob = FVMProblem(mesh, BCs;
    diffusion_function,
    diffusion_parameters,
    reaction_function=reaction,
    final_time=final_time,
    initial_condition=initial_guess,
    steady=true)
alg = DynamicSS(TRBDF2(linsolve=FastLUFactorization()))
sol = solve(prob, alg, parallel=true)
fig = Figure(fontsize=38)
ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"y")
pt_mat = Matrix(get_points(tri)')
T_mat = [T[j] for T in each_triangle(tri), j in 1:3]
msh = mesh!(ax, pt_mat, T_mat, color=sol, colorrange=(0, 10000))
Colorbar(fig[1, 2], msh)
SAVE_FIGURE && save("figures/perturbed_circle_mean_exit_time.png", fig)

## Ellipse 
using ..FiniteVolumeMethod
using DelaunayTriangulation
using CairoMakie
using OrdinaryDiffEq
using SteadyStateDiffEq
using Test
using LinearAlgebra
using StatsBase
using LinearSolve
using Krylov
include("test_setup.jl")
a = 2.0
b = 1.0
θ = LinRange(0, 2π, 100)
x = a * cos.(θ)
y = b * sin.(θ)
tri = generate_mesh(x, y, 0.2; gmsh_path=GMSH_PATH)
mesh = FVMGeometry(tri)
bc = (x, y, t, u, p) -> zero(u)
type = :D
BCs = BoundaryConditions(mesh, bc, type)
D = 2.5e-5
initial_guess = ones(num_points(tri))
initial_guess[get_boundary_nodes(tri)] .= 0.0
diffusion_function = (x, y, t, u, D) -> D
diffusion_parameters = D
reaction = (x, y, t, u, p) -> one(u)
final_time = Inf
prob = FVMProblem(mesh, BCs;
    diffusion_function,
    diffusion_parameters,
    reaction_function=reaction,
    final_time=final_time,
    initial_condition=initial_guess,
    steady=true)
sol = solve(prob, DynamicSS(TRBDF2(linsolve=KrylovJL_GMRES())))
fig = Figure(fontsize=38)
ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"y", width=600, height=300)
pt_mat = Matrix(get_points(tri)')
T_mat = [T[j] for T in each_triangle(tri), j in 1:3]
msh = mesh!(ax, pt_mat, T_mat, color=sol, colorrange=(0, 16000))
Colorbar(fig[1, 2], msh)
resize_to_layout!(fig)
SAVE_FIGURE && save("figures/ellipse_mean_exit_time.png", fig)
exact = [a^2 * b^2 / (2 * D * (a^2 + b^2)) * (1 - x^2 / a^2 - y^2 / b^2) for (x, y) in each_point(tri)]
error_fnc = (fvm_sol, exact_sol) -> 100abs(fvm_sol - exact_sol) / maximum(exact)
@test median(error_fnc.(exact, sol)) < 0.3

## Perturbed ellipse
using ..FiniteVolumeMethod
using DelaunayTriangulation
using CairoMakie
using OrdinaryDiffEq
using SteadyStateDiffEq
using Test
using LinearSolve
using Krylov
include("test_setup.jl")
a = 2.0
b = 1.0
θ = LinRange(0, 2π, 100)
ε = 1 / 20
g = θ -> sin(3θ) + cos(5θ) - sin(θ)
h = θ -> cos(3θ) + sin(5θ) - cos(θ)
x = a * (1 .+ ε .* g.(θ)) .* cos.(θ)
y = b * (1 .+ ε .* h.(θ)) .* sin.(θ)
tri = generate_mesh(x, y, 0.2; gmsh_path=GMSH_PATH)
mesh = FVMGeometry(tri)
bc = (x, y, t, u, p) -> zero(u)
type = :D
BCs = BoundaryConditions(mesh, bc, type)
D = 2.5e-5
initial_guess = zeros(num_points(tri))
diffusion_function = (x, y, t, u, D) -> D
diffusion_parameters = D
reaction = (x, y, t, u, p) -> one(u)
final_time = Inf
prob = FVMProblem(mesh, BCs;
    diffusion_function,
    diffusion_parameters,
    reaction_function=reaction,
    final_time=final_time,
    initial_condition=initial_guess,
    steady=true)
sol = solve(prob, DynamicSS(TRBDF2(linsolve=KrylovJL_GMRES())))
fig = Figure(fontsize=38)
ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"y", width=600, height=300)
pt_mat = Matrix(get_points(tri)')
T_mat = [T[j] for T in each_triangle(tri), j in 1:3]
msh = mesh!(ax, pt_mat, T_mat, color=sol, colorrange=(0, 16000))
Colorbar(fig[1, 2], msh)
resize_to_layout!(fig)
SAVE_FIGURE && save("figures/perturbed_ellipse_mean_exit_time.png", fig)

## Annulus
using ..FiniteVolumeMethod
using DelaunayTriangulation
using CairoMakie
using OrdinaryDiffEq
using SteadyStateDiffEq
using Test
using LinearAlgebra
using StatsBase
using LinearSolve
using Krylov
include("test_setup.jl")
R₁ = 2.0
R₂ = 3.0
θ = LinRange(0, 2π, 250)
x = [
    [R₂ .* cos.(θ)],
    [reverse(R₁ .* cos.(θ))] # inner boundaries are clockwise
]
y = [
    [R₂ .* sin.(θ)],
    [reverse(R₁ .* sin.(θ))] # inner boundaries are clockwise
]
tri = generate_mesh(x, y, 0.2; gmsh_path=GMSH_PATH)
mesh = FVMGeometry(tri)
outer_bc = (x, y, t, u, p) -> zero(u)
inner_bc = (x, y, t, u, p) -> zero(u)
type = [:D, :N]
BCs = BoundaryConditions(mesh, [outer_bc, inner_bc], type)
D = 6.25e-4
initial_guess = zeros(num_points(tri))
diffusion_function = (x, y, t, u, D) -> D
diffusion_parameters = D
reaction = (x, y, t, u, p) -> one(u)
final_time = Inf
prob = FVMProblem(mesh, BCs;
    diffusion_function,
    diffusion_parameters,
    reaction_function=reaction,
    final_time=final_time,
    initial_condition=initial_guess,
    steady=true)
sol = solve(prob, DynamicSS(TRBDF2(linsolve=KrylovJL_GMRES())))
fig = Figure(fontsize=38)
ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"y", width=600, height=300)
pt_mat = Matrix(get_points(tri)')
T_mat = [T[j] for T in each_triangle(tri), j in 1:3]
msh = mesh!(ax, pt_mat, T_mat, color=sol, colorrange=(0, 900))
Colorbar(fig[1, 2], msh)
resize_to_layout!(fig)
SAVE_FIGURE && save("figures/annulus_mean_exit_time.png", fig)
exact = [(R₂^2 - norm(p)^2) / (4D) + R₁^2 * log(norm(p) / R₂) / (2D) for p in each_point(tri)]
error_fnc = (fvm_sol, exact_sol) -> 100abs(fvm_sol - exact_sol) / maximum(exact)
@test median(error_fnc.(exact, sol)) < 0.1

## Perturbed annulus
using ..FiniteVolumeMethod
using DelaunayTriangulation
using CairoMakie
using OrdinaryDiffEq
using SteadyStateDiffEq
using Test
using LinearSolve
using Krylov
include("test_setup.jl")
R₁ = 2.0
R₂ = 3.0
g₁ = θ -> sin(3θ) + cos(5θ)
g₂ = θ -> cos(3θ)
θ = LinRange(0, 2π, 250)
ε = 1/20
inner_R = R₁ .* (1 .+ ε .* g₁.(θ))
outer_R = R₂ .* ( 1 .+ ε .* g₂.(θ))
x = [
    [outer_R .* cos.(θ)],
    [reverse(inner_R .* cos.(θ))] # inner boundaries are clockwise
]
y = [
    [outer_R .* sin.(θ)],
    [reverse(inner_R .* sin.(θ))] # inner boundaries are clockwise
]
tri = generate_mesh(x, y, 0.1; gmsh_path=GMSH_PATH)
mesh = FVMGeometry(tri)
outer_bc = (x, y, t, u, p) -> zero(u)
inner_bc = (x, y, t, u, p) -> zero(u)
type = [:D, :N]
BCs = BoundaryConditions(mesh, [outer_bc, inner_bc], type)
D = 6.25e-4
initial_guess = zeros(num_points(tri))
diffusion_function = (x, y, t, u, D) -> D
diffusion_parameters = D
reaction = (x, y, t, u, p) -> one(u)
final_time = Inf
prob = FVMProblem(mesh, BCs;
    diffusion_function,
    diffusion_parameters,
    reaction_function=reaction,
    final_time=final_time,
    initial_condition=initial_guess,
    steady=true)
sol = solve(prob, DynamicSS(TRBDF2(linsolve=KrylovJL_GMRES())))
fig = Figure(fontsize=38)
ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"y", width=600, height=300)
pt_mat = Matrix(get_points(tri)')
T_mat = [T[j] for T in each_triangle(tri), j in 1:3]
msh = mesh!(ax, pt_mat, T_mat, color=sol, colorrange=(0, 900))
Colorbar(fig[1, 2], msh)
resize_to_layout!(fig)
SAVE_FIGURE && save("figures/perturbed_annulus_mean_exit_time.png", fig)