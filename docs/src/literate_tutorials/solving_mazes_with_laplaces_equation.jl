# # Solving Mazes with Laplace's Equation 
# In this [tutorial](solving_mazes_with_laplaces_equation.md), we consider solving 
# mazes using Laplace's equation, applying the result of 
# [Conolly, Burns, and Weis (1990)](https://doi.org/10.1109/ROBOT.1990.126315). 
# In particular, given a maze $\mathcal M$, represented as a collection of edges together with some starting point 
# $\mathcal S_1$ and an endpoint $\mathcal S_2$, 
# Laplace's equation can be used to find the solution:
# ```math
# \begin{equation}
# \begin{aligned}
# \grad^2 \phi &= 0, & \vb x \in \mathcal M, \\
# \phi &= 0 & \vb x \in \mathcal S_1, \\
# \phi &= 1 & \vb x \in \mathcal S_2, \\
# \grad\phi\vdot\vu n &= 0 & \vb x \in \partial M \setminus (\mathcal S_1 \cup \mathcal S_2).
# \end{aligned}
# \end{equation}
# ```
# The gradient $\grad\phi$ will reveal the solution to the maze.  We just look at $\|\grad\phi\|$ 
# for revealing this solution, although other methods could e.g. use $\grad\phi$ to follow 
# the associated streamlines.
#
# ## The first maze
# Let us consider our first maze. This one will be easier, as we 
# can use only boundary conditions for it. Here is what the maze 
# looks like, where the start is in blue and the end is in red.
using DelaunayTriangulation, CairoMakie, DelimitedFiles
A = readdlm("docs/src/tutorials/maze.txt")
A = unique(A, dims=1)
x = A[1:10:end, 2] # downsample to make the problem faster
y = A[1:10:end, 1]
start = findall(y .== 648)
finish = findall(y .== 5)
start_idx_init, start_idx_end = extrema(start)
finish_idx_init, finish_idx_end = extrema(finish)
x_start = x[start]
y_start = y[start]
x_start_to_finish = [x[start_idx_end:end]; x[begin:finish_idx_init]]
y_start_to_finish = [y[start_idx_end:end]; y[begin:finish_idx_init]]
x_finish = x[finish]
y_finish = y[finish]
x_finish_to_start = x[finish_idx_end:start_idx_init]
y_finish_to_start = y[finish_idx_end:start_idx_init]
x_bnd = [x_start, x_start_to_finish, x_finish, x_finish_to_start]
y_bnd = [y_start, y_start_to_finish, y_finish, y_finish_to_start]
boundary_nodes, points = convert_boundary_points_to_indices(x_bnd, y_bnd)
tri = triangulate(points; boundary_nodes, recompute_representative_point=false) # takes a while because maze.txt contains so many points
refine!(tri)

fig, ax, sc, = triplot(tri,
    show_convex_hull=false,
    show_constrained_edges=false)
lines!(ax, [get_point(tri, get_boundary_nodes(tri, 1)...)...], color=:blue, linewidth=6)
lines!(ax, [get_point(tri, get_boundary_nodes(tri, 3)...)...], color=:red, linewidth=6)
fig

# Now we can solve the problem. 
using FiniteVolumeMethod, StableRNGs
mesh = FVMGeometry(tri)
start_bc = (x, y, t, u, p) -> zero(u)
start_to_finish_bc = (x, y, t, u, p) -> zero(u)
finish_bc = (x, y, t, u, p) -> one(u)
finish_to_start_bc = (x, y, t, u, p) -> zero(u)
fncs = (start_bc, start_to_finish_bc, finish_bc, finish_to_start_bc)
types = (Dirichlet, Neumann, Dirichlet, Neumann)
BCs = BoundaryConditions(mesh, fncs, types)
diffusion_function = (x, y, t, u, p) -> one(u)
initial_condition = 0.05randn(StableRNG(123), num_points(tri)) # random initial condition - this is the initial guess for the solution
final_time = Inf
prob = FVMProblem(mesh, BCs;
    diffusion_function=diffusion_function,
    initial_condition=initial_condition,
    final_time=final_time)
steady_prob = SteadyFVMProblem(prob)

#-
using SteadyStateDiffEq, LinearSolve, OrdinaryDiffEq
sol = solve(steady_prob, DynamicSS(TRBDF2(linsolve=KLUFactorization()), abstol=1e-14, reltol=1e-14))

# We now have our solution. 
tricontourf(tri, sol.u, colormap=:matter)

# This is not what we use to compute the solution to the maze,
# instead we need $\grad\phi$. We compute the gradient at each point using 
# NaturalNeighbours.jl. 
using NaturalNeighbours, LinearAlgebra
itp = interpolate(tri, sol.u; derivatives=true)
∇ = NaturalNeighbours.get_gradient(itp)
∇norms = norm.(∇)
tricontourf(tri, ∇norms, colormap=:matter)

# The solution to the maze is now extremely clear from this plot!

# An alternative way to look at this solution is to 
# consider the transient problem, where we do not solve the 
# steady state problem and instead view the solution over time. 
using Accessors
prob = @set prob.final_time = 1e8
LogRange(a, b, n) = exp10.(LinRange(log10(a), log10(b), n))
sol = solve(prob, TRBDF2(linsolve=KLUFactorization()), saveat=LogRange(1e2, prob.final_time, 24*10))
all_∇norms = map(sol.u) do u
    itp = interpolate(tri, u; derivatives=true)
    ∇ = NaturalNeighbours.get_gradient(itp)
    norm.(∇)
end
i = Observable(1)
∇norms = map(i -> all_∇norms[i], i)
fig, ax, sc = tricontourf(tri, ∇norms, colormap=:matter, levels=LinRange(0, 0.0035, 25), extendlow=:auto, extendhigh=:auto)
hidedecorations!(ax)
tightlimits!(ax)
fig
record(fig, joinpath(@__DIR__, "../figures", "maze_solution_1.mp4"), eachindex(sol);
    framerate=24) do _i
    i[] = _i
end;
# ```@raw html
# <figure>
#     <img src='../figures/maze_solution_1.mp4', alt='Animation of the solution to the first maze'><br>
# </figure>
# ```

# ## The second maze
# We now consider a maze that has a different form, namely the paths in the maze are made by 
# internal edges rather than from one contiguous boundary. This creates complications for us, 
# as we do not have a way to directly enforce Neumann conditions on internal edges. Instead, we must 
# represent the problem as a _differential-algebraic equation_, where the constrained equations 
# represent the Neumann conditions. 
# Before we do this, let's actually generate the maze.
using Mazes 
M = Maze(15, 25)

# This maze is an undirected graph.
M.T 

# We need to convert into something we can mesh.
using SimpleGraphs
xy = M.T.cache[:xy]
G = Mazes.Grid(M.r, M.c)
non_edge_list = [e for e in G.E if !has(M.T, e[1], e[2])]
