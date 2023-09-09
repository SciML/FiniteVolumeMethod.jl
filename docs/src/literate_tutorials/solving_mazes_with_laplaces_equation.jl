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
sol = solve(prob, TRBDF2(linsolve=KLUFactorization()), saveat=LogRange(1e2, prob.final_time, 24 * 10))
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
using Mazes, Random
Random.seed!(123) # Mazes.jl doesn't let us pass a StableRNG
M = Maze(15, 25)

# This maze is an undirected graph.
M.T

# We need to convert the graph into something we can mesh. The following 
# functions will get the internal edges and boundary points.
using SimpleGraphs
function get_internal_edges(M)
    xy = M.T.cache[:xy]
    G = Mazes.Grid(M.r, M.c)
    non_edge_list = [e for e in G.E if !has(M.T, e[1], e[2])]
    xy_edges = NTuple{2,Float64}[]
    for e in non_edge_list # taken from Maze.jl's draw
        a = e[1]
        b = e[2]
        mid = 0.5 * (xy[a] + xy[b])
        if a[1] == b[1]  # vertical segment
            x = mid[1]
            y1 = mid[2] - 0.5
            y2 = mid[2] + 0.5
            push!(xy_edges, (x, y1), (x, y2))
        else
            x1 = mid[1] - 0.5
            x2 = mid[1] + 0.5
            y = mid[2]
            push!(xy_edges, (x1, y), (x2, y))
        end
    end
    return xy_edges
end
function get_boundary_points(M)
    uℓ, ur, ℓr, ℓℓ = (1 / 2, -1 / 2), (M.c + 1 / 2, -1 / 2),
    (M.c + 1 / 2, -M.r - 1 / 2), (1 / 2, -M.r - 1 / 2)
    return uℓ, ur, ℓr, ℓℓ, uℓ
end

# In addition to these values, we also need the 
# edges representing the start and end. Here is what the maze 
# looks like: 
xy_edges, (uℓ, ur, ℓr, ℓℓ, uℓ) = get_internal_edges(M), get_boundary_points(M)
fig, ax, sc = linesegments(xy_edges)
lines!(ax, [uℓ, ur, ℓr, ℓℓ, uℓ])
fig

# The start point will be the top-left corner. Since there are 
# 15 rows, the top-left corner is at $(1/2, -1/2)$, and the point 
# below that is at $(1/2, -3/2)$:
linesegments!(ax, [(1 / 2, -1 / 2), (1 / 2, -3 / 2)], color=:blue, linewidth=6)
fig

# The ending will be the bottom-right corner, found at 
# $(25 + 1/2, -15 - 1/2)$, and the point above that is at
# $(25 + 1/2, -15 + 1/2)$:
linesegments!(ax, [(25 + 1 / 2, -15 - 1 / 2), (25 + 1 / 2, -15 + 1 / 2)], color=:red, linewidth=6)
fig

# We now define a function that returns the boundary nodes for us,
# in the form required by DelaunayTriangulation.jl.
function get_maze_boundary_nodes(M)
    uℓ, ur, ℓr, ℓℓ, uℓ = get_boundary_points(M)
    below_uℓ = uℓ .- (0, 1)
    above_ℓr = ℓr .+ (0, 1)
    boundary_nodes = [
        [uℓ, below_uℓ],
        [below_uℓ, ℓℓ, ℓr],
        [ℓr, above_ℓr],
        [above_ℓr, ur, uℓ]
    ]
    return boundary_nodes
end

# Let's now define the initial triangulation, and then think about 
# adding the edges in. 
maze_boundary_nodes = get_maze_boundary_nodes(M)
boundary_nodes, points = convert_boundary_points_to_indices(maze_boundary_nodes)
tri = triangulate(points; boundary_nodes, delete_ghosts=false)

# To add the edges in, we just use `add_edge!`, taking care to not 
# add in any duplicate points. 
xy_edges_reshape = reshape(xy_edges, 2, :)
points = get_points(tri)
for (p, q) in eachcol(xy_edges_reshape)
    p_idx = findfirst(==(p), points)
    q_idx = findfirst(==(q), points)
    if isnothing(p_idx)
        add_point!(tri, p)
        p_idx = num_points(tri)
    end
    if isnothing(q_idx)
        add_point!(tri, q)
        q_idx = num_points(tri)
    end
    add_edge!(tri, p_idx, q_idx)
end

# Finally, here is our maze.
refine!(tri; max_area=1e-3get_total_area(tri))
fig, ax, sc = triplot(tri,
    show_convex_hull=false,
    show_constrained_edges=false)
linesegments!(ax, xy_edges, color=:magenta)
linesegments!(ax, [(1 / 2, -1 / 2), (1 / 2, -3 / 2)], color=:blue, linewidth=6)
linesegments!(ax, [(25 + 1 / 2, -15 - 1 / 2), (25 + 1 / 2, -15 + 1 / 2)], color=:red, linewidth=6)
fig

# We still need to define the actual problem. Let us just 
# define what we know about the problem, and then worry about the 
# internal edges afterwards. Note that the order of the 
# boundaries is the starting point, then the edges between the start 
# and the finish, then finish point, and then the edges between 
# the finish point and the starting point.
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
final_time = 1.0
prob = FVMProblem(mesh, BCs;
    diffusion_function=diffusion_function,
    initial_condition=initial_condition,
    final_time=final_time)

# We now have our problem. But, again, we do not have any conditions 
# set at the internal edges. `InternalConditions` does not work for this, 
# as that only supports constraints at nodes. Instead, we need to
# represent the problem as a differential-algebraic equation, 
# making use of `FVMDAEProblem`. For this, we need to first 
# define a function `f!(out, du, u, p, t)` that computes the
# the residual for the (1) differential component and (2) 
# the constraint component.[^1]
# [^1]: To understand DAEs, and how we formulate them, it might help if you have 
#   read e.g. [the differential-algebraic equation example from DifferentialEquations.jl here](https://docs.sciml.ai/DiffEqDocs/stable/tutorials/dae_example/),
#   and the [DifferentialEquations.jl documentation on differential-algebraic equations](https://docs.sciml.ai/DiffEqDocs/stable/types/dae_types/).
#
# As an intermediate step, we can define a function that enforces the Neumann constraints for us. This 
# function is given by:
function neumann_constraints!(out, u, prob, t)
    tri = prob.mesh.triangulation
    cons_edges = get_constrained_edges(tri)
    for (out_idx, e) in enumerate(each_edge(cons_edges))
        i, j = e
        q = compute_flux(prob, i, j, u, t)
        out[out_idx] = q
    end
    return out
end

# This function takes in a residual vector `out` that we want equal to zero. Since we have homogeneous Neumann
# conditions at these edges, we just fill `out` with all the fluxes at each edge. The vector `u` is the current 
# solution. Now, we need to define the PDE. 
function pde_eqs!(du, u, p, t)
    n = num_points(p.prob.mesh.triangulation)
    FiniteVolumeMethod.fvm_eqs!(du, u, p, t)
    @views neumann_constraints!(du[(n+1):end], u, p.prob, t)
    return du
end

# We need to redefine the initial condition so that it is padded with zeros for each constraint. 
# We can compute the initial residuals as follows:
n = num_points(tri)
m = length(get_constrained_edges(tri))
out = zeros(m)
neumann_constraints!(out, prob.initial_condition, prob, prob.initial_time)
append!(prob.initial_condition, out)

# We are now in a position to solve the problem. We provide a mass matrix `M` so that the problem is interpreted 
# as a differential-algebraic equation.
M = zeros(n + m, n + m)
for i in 1:n
    M[i, i] = 1.0
end
sol = solve(prob, Rosenbrock23(), reltol=1e-4, mass_matrix=M, saveat=[prob.final_time], f=pde_eqs!, jac_prototype=zeros(n + m, n + m))
