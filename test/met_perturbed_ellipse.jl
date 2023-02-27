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
