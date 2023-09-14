# # Diffusion Equation on a Square Plate
# This tutorial considers a diffusion equation on a square plate:
# ```math
# \begin{equation*}
# \begin{aligned}
# \pdv{u(\vb x, t)}{t} &= \frac{1}{9}\grad^2 u(\vb x, t)  & \vb x \in \Omega,\,t>0, \\[6pt]
# u(\vb x, t) & =  0  &\vb x \in \partial\Omega,\,t>0,\\[6pt]
# u(\vb x, 0) &= f(\vb x) & \vb x \in \Omega,
# \end{aligned}
# \end{equation*}
# ```
# where $\Omega = [0, 2]^2$ and
# ```math
# f(x, y) = \begin{cases} 50 & y \leq 1, \\ 0 & y > 1. \end{cases}
# ```
# To solve this problem, the first step is to define the mesh.
using FiniteVolumeMethod, DelaunayTriangulation
using Test #src
using ReferenceTests #src
using StatsBase #src
a, b, c, d = 0.0, 2.0, 0.0, 2.0
nx, ny = 50, 50
tri = triangulate_rectangle(a, b, c, d, nx, ny, single_boundary=true)
mesh = FVMGeometry(tri)

# This mesh is shown below.
using CairoMakie
fig, ax, sc = triplot(tri)
fig

# We now need to define the boundary conditions. We have a homogeneous Dirichlet condition:
bc = (x, y, t, u, p) -> zero(u)
BCs = BoundaryConditions(mesh, bc, Dirichlet)

# We can now define the actual PDE. We start by defining the initial condition and the diffusion function. 
f = (x, y) -> y ≤ 1.0 ? 50.0 : 0.0
initial_condition = [f(x, y) for (x, y) in each_point(tri)]
D = (x, y, t, u, p) -> 1 / 9

# We can now define the problem:
final_time = 0.5
prob = FVMProblem(mesh, BCs; diffusion_function=D, initial_condition, final_time)

# Note that in `prob`, it is not a diffusion function that is used but instead it is a flux function:
prob.flux_function

# When providing `diffusion_function`, the flux is given by $\vb q(\vb x, t, \alpha,\beta,\gamma) = (-\alpha/9, -\beta/9)^{\mkern-1.5mu\mathsf{T}}$,
# where $(\alpha, \beta, \gamma)$ defines the approximation to $u$ via $u(x, y) = \alpha x + \beta y + \gamma$ so that 
# $\grad u(\vb x, t) = (\alpha,\beta)^{\mkern-1.5mu\mathsf{T}}$.

# To now solve the problem, we simply use `solve`. When no algorithm 
# is provided, as long as DifferentialEquations is loaded (instead of e.g. 
# OrdinaryDiffEq), the algorithm is chosen automatically. Moreover, note that, 
# in the `solve` call below, multithreading is enabled by default.
using DifferentialEquations
sol = solve(prob, saveat=0.05)

# To visualise the solution, we can use `tricontourf!` from Makie.jl. 
fig = Figure(fontsize=38)
for (i, j) in zip(1:3, (1, 6, 11))
    ax = Axis(fig[1, i], width=600, height=600,
        xlabel="x", ylabel="y",
        title="t = $(sol.t[j])",
        titlealign=:left)
    tricontourf!(ax, tri, sol.u[j], levels=0:5:50, colormap=:matter)
    tightlimits!(ax)
end
resize_to_layout!(fig)
fig
@test_reference joinpath(@__DIR__, "../figures", "diffusion_equation_on_a_square_plate.png") fig #src

function exact_solution(x, y, t) #src
    if t > 0 #src
        s = 0.0 #src
        for m in 1:2:50 #src
            mterm = 2 / m * sin(m * π * x / 2) * exp(-π^2 * m^2 * t / 36) #src
            for n in 1:50 #src
                nterm = (1 - cos(n * π / 2)) / n * sin(n * π * y / 2) * exp(-π^2 * n^2 * t / 36) #src
                s += mterm * nterm #src
            end #src
        end #src
        return 200s / π^2 #src
    else #src
        return y ≤ 1.0 ? 50.0 : 0.0 #src
    end #src
end #src
function compare_solutions(sol, tri) #src
    n = DelaunayTriangulation.num_solid_vertices(tri) #src
    x = zeros(n, length(sol)) #src
    y = zeros(n, length(sol)) #src
    u = zeros(n, length(sol)) #src
    for i in eachindex(sol) #src
        for j in each_solid_vertex(tri) #src
            x[j, i], y[j, i] = get_point(tri, j) #src
            u[j, i] = exact_solution(x[j, i], y[j, i], sol.t[i]) #src
        end #src
    end #src
    return x, y, u #src
end #src
x, y, u = compare_solutions(sol, tri) #src
fig = Figure(fontsize=64) #src
for i in eachindex(sol) #src
    ax = Axis(fig[1, i], width=600, height=600) #src
    tricontourf!(ax, tri, sol.u[i], levels=0:5:50, colormap=:matter) #src
    ax = Axis(fig[2, i], width=600, height=600) #src
    tricontourf!(ax, tri, u[:, i], levels=0:5:50, colormap=:matter) #src
end #src
resize_to_layout!(fig) #src
fig #src
@test_reference joinpath(@__DIR__, "../figures", "diffusion_equation_on_a_square_plate_exact_comparisons.png") fig #src