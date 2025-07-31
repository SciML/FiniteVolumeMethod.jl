using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # Reaction-Diffusion Equation with a Time-dependent Dirichlet Boundary Condition on a Disk 
# In this tutorial, we consider a reaction-diffusion equation 
# on a disk with a boundary condition of the form $\mathrm du/\mathrm dt = u$:
# ```math
# \begin{equation*}
# \begin{aligned}
# \pdv{u(r, \theta, t)}{t} &= \div[u\grad u] + u(1-u) & 0<r<1,\,0<\theta<2\pi,\\[6pt]
# \dv{u(1, \theta, t)}{t} &= u(1,\theta,t) & 0<\theta<2\pi,\,t>0,\\[6pt]
# u(r,\theta,0) &= \sqrt{I_0(\sqrt{2}r)} & 0<r<1,\,0<\theta<2\pi,
# \end{aligned}
# \end{equation*}
# ```
# where $I_0$ is the modified Bessel function of the first kind of order zero. 
# For this problem the diffusion function is $D(\vb x, t, u) = u$ and the source function 
# is $R(\vb x, t, u) = u(1-u)$, or equivalently the force function is 
# ```math
# \vb q(\vb x, t, \alpha,\beta,\gamma) = \left(-\alpha(\alpha x + \beta y + \gamma), -\beta(\alpha x + \beta y + \gamma)\right)^{\mkern-1.5mu\mathsf{T}}.
# ```
# As usual, we start by generating the mesh.
using FiniteVolumeMethod, DelaunayTriangulation, ElasticArrays
r = 1.0
circle = CircularArc((0.0, r), (0.0, r), (0.0, 0.0))
points = NTuple{2, Float64}[]
boundary_nodes = [circle]
tri = triangulate(points; boundary_nodes)
A = get_area(tri)
refine!(tri; max_area = 1e-4A)
mesh = FVMGeometry(tri)

#-
using CairoMakie
triplot(tri)

# Now we define the boundary conditions and the PDE.
using Bessels
BCs = BoundaryConditions(mesh, (x, y, t, u, p) -> u, Dudt)

#-
f = (x, y) -> sqrt(besseli(0.0, sqrt(2) * sqrt(x^2 + y^2)))
D = (x, y, t, u, p) -> u
R = (x, y, t, u, p) -> u * (1 - u)
initial_condition = [f(x, y) for (x, y) in DelaunayTriangulation.each_point(tri)]
final_time = 0.10
prob = FVMProblem(mesh, BCs;
    diffusion_function = D,
    source_function = R,
    final_time,
    initial_condition)

# We can now solve. 
using OrdinaryDiffEq, LinearSolve
alg = FBDF(linsolve = UMFPACKFactorization(), autodiff = false)
sol = solve(prob, alg, saveat = 0.01)
sol |> tc #hide

#-
using ReferenceTests #src
fig = Figure(fontsize = 38)
for (i, j) in zip(1:3, (1, 6, 11))
    ax = Axis(fig[1, i], width = 600, height = 600,
        xlabel = "x", ylabel = "y",
        title = "t = $(sol.t[j])",
        titlealign = :left)
    tricontourf!(ax, tri, sol.u[j], levels = 1:0.01:1.4, colormap = :matter)
    tightlimits!(ax)
end
resize_to_layout!(fig)
fig
@test_reference joinpath(@__DIR__,
    "../figures",
    "reaction_diffusion_equation_with_a_time_dependent_dirichlet_boundary_condition_on_a_disk.png") fig #src

using ReferenceTests #src
function exact_solution(x, y, t) #src
    r = sqrt(x^2 + y^2) #src
    return exp(t) * sqrt(besseli(0.0, sqrt(2) * r)) #src
end #src
function compare_solutions(sol, tri) #src
    n = DelaunayTriangulation.num_points(tri) #src
    x = zeros(n, length(sol)) #src
    y = zeros(n, length(sol)) #src
    u = zeros(n, length(sol)) #src
    for i in eachindex(sol) #src
        !DelaunayTriangulation.has_vertex(tri, i) && continue #src
        for j in each_solid_vertex(tri) #src
            x[j, i], y[j, i] = get_point(tri, j) #src
            u[j, i] = exact_solution(x[j, i], y[j, i], sol.t[i]) #src
        end #src
    end #src
    return x, y, u #src
end #src
x, y, u = compare_solutions(sol, tri) #src
fig = Figure(fontsize = 64) #src
for i in eachindex(sol) #src
    ax = Axis(fig[1, i], width = 600, height = 600) #src
    tricontourf!(ax, tri, sol.u[i], levels = 1:0.01:1.4, colormap = :matter) #src
    ax = Axis(fig[2, i], width = 600, height = 600) #src
    tricontourf!(ax, tri, u[:, i], levels = 1:0.01:1.4, colormap = :matter) #src
end #src
resize_to_layout!(fig) #src
fig #src
@test_reference joinpath(@__DIR__,
    "../figures",
    "reaction_diffusion_equation_with_a_time_dependent_dirichlet_boundary_condition_on_a_disk_exact_comparisons.png") fig #src
