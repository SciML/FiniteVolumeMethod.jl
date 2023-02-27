using ..FiniteVolumeMethod
include("test_setup.jl")
using CairoMakie
using LinearSolve
using OrdinaryDiffEq
using Test

## Generate the mesh. 
# When specifying multiple boundary curves, the outer boundary comes first 
# and is given in counter-clockwise order. The inner boundaries then follow. 
R₁ = 0.2
R₂ = 1.0
θ = LinRange(0, 2π, 100)
x = [
    [R₂ .* cos.(θ)],
    [reverse(R₁ .* cos.(θ))]
]
y = [
    [R₂ .* sin.(θ)],
    [reverse(R₁ .* sin.(θ))]
]
tri = generate_mesh(x, y, 0.2; gmsh_path=GMSH_PATH)
mesh = FVMGeometry(tri)

## Define the boundary conditions 
outer_bc = (x, y, t, u, p) -> zero(u)
inner_bc = (x, y, t, u, p) -> 50.0 * (1.0 - exp(-0.5t))
type = [:N, :D]
BCs = BoundaryConditions(mesh, [outer_bc, inner_bc], type)

## Define the problem 
initial_condition_f = (x, y) -> begin
    10 * exp(-25 * ((x + 0.5) * (x + 0.5) + (y + 0.5) * (y + 0.5))) - 5 * exp(-50 * ((x + 0.3) * (x + 0.3) + (y + 0.5) * (y + 0.5))) - 10 * exp(-45 * ((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)))
end
diffusion = (x, y, t, u, p) -> one(u)
points = get_points(tri)
u₀ = @views initial_condition_f.(points[1, :], points[2, :])
final_time = 2.0
prob = FVMProblem(mesh, BCs;
    diffusion_function=diffusion,
    final_time=final_time,
    initial_condition=u₀)

## Solve the problem 
alg = TRBDF2(linsolve=KLUFactorization())
sol = solve(prob, alg, parallel=false, saveat=0.2)

## Visualise 
pt_mat = Matrix(points')
T_mat = [T[j] for T in each_triangle(tri), j in 1:3]
fig = Figure(resolution=(2068.72f0, 686.64f0), fontsize=38)
ax = Axis(fig[1, 1], width=600, height=600)
mesh!(ax, pt_mat, T_mat, color=sol.u[1], colorrange=(-10, 20), colormap=:viridis)
ax = Axis(fig[1, 2], width=600, height=600)
mesh!(ax, pt_mat, T_mat, color=sol.u[6], colorrange=(-10, 20), colormap=:viridis)
ax = Axis(fig[1, 3], width=600, height=600)
mesh!(ax, pt_mat, T_mat, color=sol.u[11], colorrange=(-10, 20), colormap=:viridis)
SAVE_FIGURE && save("figures/annulus_test.png", fig)

if SAVE_FIGURE
    fig = Figure()
    t_rng = LinRange(0, 2, 361)
    j = Observable(1)
    ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"y",
        title=Makie.lift(_j -> L"t = %$(rpad(round(t_rng[_j], digits = 5), 7, '0'))", j),
        titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=@lift(sol(t_rng[$j])), colorrange=(-10, 20), colormap=:viridis)
    record(fig, "figures/annulus_test.mp4", eachindex(t_rng)) do _j
        j[] = _j
    end
end