# just make sure everything still works as advertised
using ..FiniteVolumeMethod, DelaunayTriangulation, CairoMakie, OrdinaryDiffEq
using ReferenceTests
a, b, c, d = 0.0, 2.0, 0.0, 2.0
nx, ny = 50, 50
tri = triangulate_rectangle(a, b, c, d, nx, ny, single_boundary=true)
mesh = FVMGeometry(tri)
bc = (x, y, t, u, p) -> zero(u)
BCs = BoundaryConditions(mesh, bc, Dirichlet)
f = (x, y) -> y ≤ 1.0 ? 50.0 : 0.0
initial_condition = [f(x, y) for (x, y) in DelaunayTriangulation.each_point(tri)]
D = (x, y, t, u, p) -> 1 / 9
final_time = 0.5
prob = FVMProblem(mesh, BCs; diffusion_function=D, initial_condition, final_time)
sol = solve(prob, Tsit5(), saveat=0.001)
u = Observable(sol.u[421])
fig, ax, sc = tricontourf(tri, u, levels=0:5:50, colormap=:matter)
tightlimits!(ax)
fig 
@test_reference "test_figures/readme_example.png" fig