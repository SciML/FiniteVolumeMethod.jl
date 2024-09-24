using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # Piecewise Linear and Natural Neighbour Inteprolation for an Advection-Diffusion Equation 
# In this tutorial, we have three aims:
#
# 1. Demonstrate how to solve an advection-diffusion equation. 
# 2. Demonstrate how piecewise linear interpolation can be applied to a PDE solution at each time.
# 3. Demonstrate how  [NaturalNeighbours.jl](https://github.com/DanielVandH/NaturalNeighbours.jl) can be applied to compute more accurate interpolants than piecewise linear interpolation at each time. 
#
# The equation we will be considering is 
# ```math
# \begin{equation}\label{eq:advdiffeq}
# \begin{aligned}
# \pdv{u}{t} &= D\pdv[2]{u}{x} + D\pdv[2]{u}{y} - \nu\pdv{u}{x},
# \end{aligned}
# \end{equation}
# ```
# with $u(\vb x, 0) = \delta(\vb x)$ and homogeneous Dirichlet conditions, 
# where $\delta$ is the Dirac delta function. This equation is defined on 
# $\mathbb R^2$, but we will replace $\mathbb R^2$ with $\Omega = [-L, L]^2$ 
# for $L = 30$.

# ## Solving the problem 
# We start by defining and solving the problem associated 
# with \eqref{eq:advdiffeq}. For the mesh, we could use 
# `triangulate_rectangle`, but we want to put most of the triangles 
# near the origin, so we need to use `refine!` on an initial mesh.
using DelaunayTriangulation, FiniteVolumeMethod, LinearAlgebra, CairoMakie
L = 30
tri = triangulate_rectangle(-L, L, -L, L, 2, 2, single_boundary=true)
tot_area = get_area(tri)
max_area_function = (A, r) -> 1e-6tot_area * r^2 / A
area_constraint = (_tri, T) -> begin
    u, v, w = triangle_vertices(T)
    p, q, r = get_point(_tri, u, v, w)
    c = (p .+ q .+ r) ./ 3
    dist_to_origin = norm(c)
    A = DelaunayTriangulation.triangle_area(p, q, r)
    flag = A ≥ max_area_function(A, dist_to_origin)
    return flag
end
refine!(tri; min_angle=33.0, custom_constraint=area_constraint)
triplot(tri)

#-
mesh = FVMGeometry(tri)

# The boundary conditions are homogeneous `Dirichlet` conditions. 
BCs = BoundaryConditions(mesh, (x, y, t, u, p) -> zero(u), Dirichlet)

# We now need to define the actual problem. We need to write \eqref{eq:advdiffeq}
# in the form 
# ```math
# \pdv{u}{t} + \div\vb q = 0.
# ```
# To do this, write:
# ```math 
# \begin{align*}
# \div \vb q &= \nu\pdv{u}{x} - D\pdv[2]{u}{x} - D\pdv[2]{u}{y} \\
# &= \pdv{x}\left(\nu u - D\pdv{u}{x}\right) - \pdv{y}\left(D\pdv{u}{y}\right) \\
# &= \div \begin{bmatrix} \nu u - D\pdv{u}{x} & -D\pdv{u}{y} \end{bmatrix}^{\mkern-1.5mu\mathsf{T}} \\
# &= \div \left(\boldsymbol\nu u - D\grad u\right),
# \end{align*}
# ```
# where $\boldsymbol\nu = (\nu, 0)^{\mkern-1.5mu\mathsf{T}}$. Thus, we can write
# ```math
# \vb q = \boldsymbol\nu u - D\grad u. 
# ```
# We now have our flux function. Next, let us define the initial condition. 
# We approximate by 
# ```math
# \delta(\vb x) \approx g(\vb x) \approx \frac{1}{\varepsilon^2\pi}\exp\left[-\frac{1}{\varepsilon^2}\left(x^2+y^2\right)\right],
# ```
# taking $\varepsilon=1/10$. We can now define the problem. Remember that the flux function 
# takes argument $(\alpha, \beta, \gamma)$ rather than $u$, replacing $u$ with $u(x, y) = \alpha x + \beta y + \gamma$,
# and it returns a `Tuple` representing the vector. We let $D = 0.02$ and $\nu = 0.05$.
ε = 1 / 10
f = (x, y) -> 1 / (ε^2 * π) * exp(-(x^2 + y^2) / ε^2)
initial_condition = [f(x, y) for (x, y) in DelaunayTriangulation.each_point(tri)]
flux_function = (x, y, t, α, β, γ, p) -> begin
    ∂x = α
    ∂y = β
    u = α * x + β * y + γ
    qx = p.ν * u - p.D * ∂x
    qy = -p.D * ∂y
    return (qx, qy)
end
flux_parameters = (D=0.02, ν=0.05)
final_time = 250.0
prob = FVMProblem(mesh, BCs;
    initial_condition,
    flux_function,
    flux_parameters,
    final_time)

# Now we can solve and visualise the solution.
using OrdinaryDiffEq, LinearSolve
times = [0, 10, 25, 50, 100, 200, 250]
sol = solve(prob, TRBDF2(linsolve=KLUFactorization()), saveat=times)
sol |> tc #hide

#-
using CairoMakie
using ReferenceTests #src
fig = Figure(fontsize=38)
for i in eachindex(sol)
    ax = Axis(fig[1, i], width=400, height=400,
        xlabel="x", ylabel="y",
        title="t = $(sol.t[i])",
        titlealign=:left)
    tricontourf!(ax, tri, sol.u[i], levels=0:0.00001:0.001, extendhigh=:auto, extendlow=:auto, colormap=:matter)
    tightlimits!(ax)
    ylims!(ax, -10, 10)
end
resize_to_layout!(fig)
fig
@test_reference joinpath(@__DIR__, "../figures", "piecewise_linear_and_natural_neighbour_interpolation_for_an_advection_diffusion_equation.png") fig #src

using StatsBase, Test #src
_sol = solve(prob, TRBDF2(linsolve=KLUFactorization()), saveat=1.0) #src
function exact_solution(x, y, t, D, ν) #src
    return 1 / (4 * D * π * t) * exp(1 / (4 * D * t) * (-(x - ν * t)^2 - y^2)) #src
end #src
function get_errs(_sol, tri, flux_parameters) #src
    _errs = zeros(length(_sol)) #src
    _err = zeros(DelaunayTriangulation.num_points(tri)) #src
    for i in eachindex(_sol) #src
        !DelaunayTriangulation.has_vertex(tri, i) && continue #src
        i == 1 && continue #src
        m = maximum(_sol.u[i]) #src
        for j in each_solid_vertex(tri) #src
            e = exact_solution(get_point(tri, j)..., _sol.t[i], flux_parameters.D, flux_parameters.ν) #src
            _err[j] = 100abs((_sol.u[i][j] - e) / m) #src
        end #src
        _errs[i] = mean(_err) #src
    end #src
    return _errs #src
end #src
_errs = get_errs(_sol, tri, flux_parameters) #src
@test all(≤(0.16), _errs) #src
fig = Figure(fontsize=64) #src
t = [10, 25, 50, 100, 200, 250] #src
t_idx = [findlast(≤(τ), _sol.t) for τ in t] #src
for (i, j) in enumerate(t_idx) #src
    ax = Axis(fig[1, i], width=400, height=400) #src
    tricontourf!(ax, tri, _sol.u[j], levels=0:0.00001:0.001, extendhigh=:auto, extendlow=:auto, colormap=:matter) #src
    ax = Axis(fig[2, i], width=400, height=400) #src
    tricontourf!(ax, tri, [exact_solution(x, y, _sol.t[j], flux_parameters.D, flux_parameters.ν) for (x, y) in DelaunayTriangulation.each_point(tri)], levels=0:0.00001:0.001, extendhigh=:auto, extendlow=:auto, colormap=:matter) #src
end #src
resize_to_layout!(fig) #src
fig #src
@test_reference joinpath(@__DIR__, "../figures", "piecewise_linear_and_natural_neighbour_interpolation_for_an_advection_diffusion_equation_exact_comparisons.png") fig #src

# ## Piecewise linear interpolation
# As mentioned in [mathematical details section](../math.md), a key part of the finite volume method is the assumption that 
# $u$ is piecewise linear between each triangular element, letting $u(x, y) = \alpha x + \beta y + \gamma$. Thus, 
# it may be natural to want to interpolate the solution using piecewise linear interpolation. This could be done 
# by making use of `jump_and_march` from DelaunayTriangulation.jl to find the triangle containing a given point 
# $(x, y)$ and then use `pl_interpolate` to interpolate the solution at the point; we do not provide a method 
# that gets this triangle for you and then interpolates without this intermediate `jump_and_march`, 
# as it is typically more efficient to first obtain all the triangles you need 
# and then interpolate. In what follows, we:
#
# 1. Define a grid to interpolate over.
# 2. Find the triangles containing each point in the grid.
# 3. Interpolate at each point for the given times. 
#
# We consider the times $t = 10, 25, 50, 100, 200, 250$. You could also of course
# amend the procedure so that you evaluate the interpolant at each time for a given point first, 
# allowing you to avoid storing the triangle since you only consider each point a single time.
x = LinRange(-L, L, 250)
y = LinRange(-L, L, 250)
triangles = Matrix{NTuple{3,Int}}(undef, length(x), length(y))
for j in eachindex(y)
    for i in eachindex(x)
        triangles[i, j] = jump_and_march(tri, (x[i], y[j]))
    end
end
interpolated_vals = zeros(length(x), length(y), length(sol))
for k in eachindex(sol)
    for j in eachindex(y)
        for i in eachindex(x)
            interpolated_vals[i, j, k] = pl_interpolate(prob, triangles[i, j], sol.u[k], x[i], y[j])
        end
    end
end

# Let's visualise these results to check their accuracy. We compute the triangulation of 
# our grid to make the `tricontourf` call faster.
_tri = triangulate([[x for x in x, _ in y] |> vec [y for _ in x, y in y] |> vec]')
fig = Figure(fontsize=38)
for i in eachindex(sol)
    ax = Axis(fig[1, i], width=400, height=400,
        xlabel="x", ylabel="y",
        title="t = $(sol.t[i])",
        titlealign=:left)
    tricontourf!(ax, _tri, interpolated_vals[:, :, i] |> vec, levels=0:0.00001:0.001, extendhigh=:auto, extendlow=:auto, colormap=:matter)
    tightlimits!(ax)
    ylims!(ax, -10, 10)
end
resize_to_layout!(fig)
fig
@test_reference joinpath(@__DIR__, "../figures", "piecewise_linear_and_natural_neighbour_interpolation_for_an_advection_diffusion_equation_piecewise_linear_interpolation.png") fig #src

# ## Natural neighbour interpolation 
# Since the solution is defined over a triangulation, the most natural form of inteprolation to use, 
# other than piecewise linear interpolation, is natural neighbour interpolation. We can use
# [NaturalNeighbours.jl](https://github.com/DanielVandH/NaturalNeighbours.jl) for this; 
# NaturalNeighbours.jl also provides the same piecewise linear interpolant above via its 
# `Triangle()` interpolator, which may be more efficient as it has multithreading built in. 
#
# The way to construct a natural neighbour interpolant is as follows, where we provide 
# the interpolant with the solution at $t = 50$.
using NaturalNeighbours
itp = interpolate(tri, sol.u[4], derivatives=true) # sol.t[4] == 50
sol |> tc #hide

# We need `derivatives = true` so that we can use the higher order interpolants `Sibson(1)`, `Hiyoshi(2)`,
# and `Farin()` below - if you don't use those, then you shouldn't need this option (unless you 
# want to later differentiate the interpolant using `differentiate`, then yes you do need it).

# We can then evaluate this interpolant by simply calling it. The most efficient 
# way to call it is by providing it with a vector of points, rather than broadcasting 
# over points, since multithreading can be used in this case. Let us 
# interpolate at the grid from before, which requires us to collect it into a vector:
_x = [x for x in x, _ in y] |> vec
_y = [y for _ in x, y in y] |> vec;

# We will look at all the interpolants provided by NaturalNeighbours.jl.[^1]

# [^1]: This list is available from `?NaturalNeighbours.AbstractInterpolator`. Look at the help page (`?`) for the respective interpolators or NaturalNeighbours.jl's documentation for more information.
sibson_vals = itp(_x, _y; method=Sibson())
triangle_vals = itp(_x, _y; method=Triangle()) # this is the same as pl_interpolate
laplace_vals = itp(_x, _y; method=Laplace())
sibson_1_vals = itp(_x, _y; method=Sibson(1))
nearest_vals = itp(_x, _y; method=Nearest())
farin_vals = itp(_x, _y; method=Farin())
hiyoshi_vals = itp(_x, _y; method=Hiyoshi(2))
pde_vals = sol.u[4];

# We visualise these results as follows.
fig = Figure(fontsize=38)
all_vals = (sibson_vals, triangle_vals, laplace_vals, sibson_1_vals, nearest_vals, farin_vals, hiyoshi_vals, pde_vals)
titles = ("(a): Sibson", "(b): Triangle", "(c): Laplace", "(d): Sibson-1", "(e): Nearest", "(f): Farin", "(g): Hiyoshi", "(h): PDE")
fig = Figure(fontsize=55, resolution=(6350, 1550)) # resolution from resize_to_layout!(fig) - had to manually adjust to fix missing ticks
for (i, (vals, title)) in enumerate(zip(all_vals, titles))
    ax2d = Axis(fig[1, i], xlabel="x", ylabel="y", width=600, height=600, title=title, titlealign=:left)
    ax3d = Axis3(fig[2, i], xlabel="x", ylabel="y", width=600, height=600, title=title, titlealign=:left)
    ax3d.zlabeloffset[] = 125
    xlims!(ax2d, -4, 6)
    ylims!(ax2d, -4, 4)
    xlims!(ax3d, -4, 6)
    ylims!(ax3d, -4, 4)
    if vals ≠ pde_vals
        contourf!(ax2d, _x, _y, vals, colormap=:matter, levels=0:0.001:0.1, extendlow=:auto, extendhigh=:auto)
        vals = copy(vals)
        vals[(_x.<-4).|(_x.>6)] .= NaN
        vals[(_y.<-4).|(_y.>4)] .= NaN # This is the only way to fix the weird issues with Axis3 when changing the (x/y/z)lims...
        surface!(ax3d, _x, _y, vals, color=vals, colormap=:matter, colorrange=(0, 0.1))
    else
        tricontourf!(ax2d, tri, vals, colormap=:matter, levels=0:0.001:0.1, extendlow=:auto, extendhigh=:auto)
        triangles = [T[j] for T in each_solid_triangle(tri), j in 1:3]
        x = getx.(get_points(tri))
        y = gety.(get_points(tri))
        vals = copy(vals)
        vals[(x.<-4).|(x.>6)] .= NaN
        vals[(y.<-4).|(y.>4)] .= NaN # This is the only way to fix the weird issues with Axis3 when changing the (x/y/z)lims...
        mesh!(ax3d, hcat(x, y, vals), triangles, color=vals, colormap=:matter, colorrange=(0, 0.1))
    end
end
fig
@test_reference joinpath(@__DIR__, "../figures", "piecewise_linear_and_natural_neighbour_interpolation_for_an_advection_diffusion_equation_natural_neighbour_interpolation.png") fig #src

# We note that natural neighbour interpolation is not technically well defined 
# for constrained triangulations. In this case it is fine, but for regions 
# with, say, holes or non-convex boundaries, you may run into issues. For such 
# cases, you should usually call the interpolant with `project=false` to at least 
# help the procedure a bit. You may also be interested in `identify_exterior_points`.
# We consider interpolating data over a region with holes in [this annulus example](diffusion_equation_on_an_annulus.md).