```@meta
EditURL = "https://github.com/SciML/FiniteVolumeMethod.jl/tree/main/docs/src/literate_tutorials/diffusion_equation_on_an_annulus.jl"
```

````@example diffusion_equation_on_an_annulus
using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
nothing #hide
````

# Diffusion Equation on an Annulus
In this tutorial, we consider a
diffusion equation on an annulus:
```math
\begin{equation}
\begin{aligned}
\pdv{u(\vb x, t)}{t} &= \grad^2 u(\vb x, t) & \vb x \in \Omega, \\
\grad u(\vb x, t) \vdot \vu n(\vb x) &= 0 & \vb x \in \mathcal D(0, 1), \\
u(\vb x, t) &= c(t) & \vb x \in \mathcal D(0,0.2), \\
u(\vb x, t) &= u_0(\vb x),
\end{aligned}
\end{equation}
```
demonstrating how we can solve PDEs over multiply-connected domains.
Here, $\mathcal D(0, r)$ is a circle of radius $r$ centred at the origin,
$\Omega$ is the annulus between $\mathcal D(0,0.2)$ and
$\mathcal D(0, 1)$, $c(t) = 50[1-\mathrm{e}^{-t/2}]$, and
```math
u_0(x) = 10\mathrm{e}^{-25\left[\left(x+\frac12\right)^2+\left(y+\frac12\right)^2\right]} - 10\mathrm{e}^{-45\left[\left(x-\frac12\right)^2+\left(y-\frac12\right)^2\right]} - 5\mathrm{e}^{-50\left[\left(x+\frac{3}{10}\right)^2+\left(y+\frac12\right)^2\right]}.
```
For the mesh, we use two `CircularArc`s to define the annulus.

````@example diffusion_equation_on_an_annulus
using DelaunayTriangulation, FiniteVolumeMethod, CairoMakie
R₁, R₂ = 0.2, 1.0
inner = CircularArc((R₁, 0.0), (R₁, 0.0), (0.0, 0.0), positive=false)
outer = CircularArc((R₂, 0.0), (R₂, 0.0), (0.0, 0.0))
boundary_nodes = [[[outer]], [[inner]]]
points = NTuple{2,Float64}[]
tri = triangulate(points; boundary_nodes)
A = get_area(tri)
refine!(tri; max_area=1e-4A)
triplot(tri)
````

````@example diffusion_equation_on_an_annulus
mesh = FVMGeometry(tri)
````

Now let us define the boundary conditions. Remember,
the order of the boundary conditions follows the order
of the boundaries in the mesh. The outer boundary
came first, and then came the inner boundary. We can verify
that this is the order of the boundary indices as
follows:

````@example diffusion_equation_on_an_annulus
fig = Figure()
ax = Axis(fig[1, 1])
outer = [get_point(tri, i) for i in get_neighbours(tri, -1)]
inner = [get_point(tri, i) for i in get_neighbours(tri, -2)]
triplot!(ax, tri)
scatter!(ax, outer, color=:red)
scatter!(ax, inner, color=:blue)
fig
````

So, the boundary conditions are:

````@example diffusion_equation_on_an_annulus
outer_bc = (x, y, t, u, p) -> zero(u)
inner_bc = (x, y, t, u, p) -> oftype(u, 50(1 - exp(-t / 2)))
types = (Neumann, Dirichlet)
BCs = BoundaryConditions(mesh, (outer_bc, inner_bc), types)
````

Finally, let's define the problem and solve it.

````@example diffusion_equation_on_an_annulus
initial_condition_f = (x, y) -> begin
    10 * exp(-25 * ((x + 0.5) * (x + 0.5) + (y + 0.5) * (y + 0.5))) - 5 * exp(-50 * ((x + 0.3) * (x + 0.3) + (y + 0.5) * (y + 0.5))) - 10 * exp(-45 * ((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)))
end
diffusion_function = (x, y, t, u, p) -> one(u)
initial_condition = [initial_condition_f(x, y) for (x, y) in DelaunayTriangulation.each_point(tri)]
final_time = 2.0
prob = FVMProblem(mesh, BCs;
    diffusion_function,
    final_time,
    initial_condition)
````

````@example diffusion_equation_on_an_annulus
using OrdinaryDiffEq, LinearSolve
sol = solve(prob, TRBDF2(linsolve=KLUFactorization()), saveat=0.2)
sol |> tc #hide
````

````@example diffusion_equation_on_an_annulus
fig = Figure(fontsize=38)
for (i, j) in zip(1:3, (1, 6, 11))
    local ax
    ax = Axis(fig[1, i], width=600, height=600,
        xlabel="x", ylabel="y",
        title="t = $(sol.t[j])",
        titlealign=:left)
    tricontourf!(ax, tri, sol.u[j], levels=-10:2:40, colormap=:matter)
    tightlimits!(ax)
end
resize_to_layout!(fig)
fig
````

To finish this example, let us consider how
natural neighbour interpolation can be applied here. The
application is more complicated for this problem since
the mesh has holes. Before we do that, though, let us
show how we could use `pl_interpolate`, which could
be useful if we did not need a higher quality interpolant.
Let us interpolate the solution at $t = 1$, which
is `sol.t[6]`. For this, we need to put the ghost
triangles back into `tri` so that we can safely
apply `jump_and_march`. This is done with `add_ghost_triangles!`.

````@example diffusion_equation_on_an_annulus
add_ghost_triangles!(tri)
````

(Actually, `tri` already had these ghost triangles,
but we are just showing how you would add them back in if needed.)

Now let's interpolate.

````@example diffusion_equation_on_an_annulus
x = LinRange(-R₂, R₂, 400)
y = LinRange(-R₂, R₂, 400)
interp_vals = zeros(length(x), length(y))
u = sol.u[6]
last_triangle = Ref((1, 1, 1))
for (j, _y) in enumerate(y)
    for (i, _x) in enumerate(x)
        T = jump_and_march(tri, (_x, _y), try_points=last_triangle[])
        last_triangle[] = triangle_vertices(T) # used to accelerate jump_and_march, since the points we're looking for are close to each other
        if DelaunayTriangulation.is_ghost_triangle(T) # don't extrapolate
            interp_vals[i, j] = NaN
        else
            interp_vals[i, j] = pl_interpolate(prob, T, sol.u[6], _x, _y)
        end
    end
end
fig, ax, sc = contourf(x, y, interp_vals, levels=-10:2:40, colormap=:matter)
fig
````

Let's now consider applying NaturalNeighbours.jl. We apply it naively first to
highlight some complications.

````@example diffusion_equation_on_an_annulus
using NaturalNeighbours
_x = vec([x for x in x, y in y]) # NaturalNeighbours.jl needs vector data
_y = vec([y for x in x, y in y])
itp = interpolate(tri, u, derivatives=true)
itp |> tc #hide
````

````@example diffusion_equation_on_an_annulus
itp_vals = itp(_x, _y; method=Farin())
itp_vals |> tc #hide
````

````@example diffusion_equation_on_an_annulus
fig, ax, sc = contourf(x, y, reshape(itp_vals, length(x), length(y)), colormap=:matter, levels=-10:2:40)
fig
````

The issue here is that the interpolant is trying to extrapolate inside the hole and
outside of the annulus. To avoid this, you need to pass `project=false`.

````@example diffusion_equation_on_an_annulus
itp_vals = itp(_x, _y; method=Farin(), project=false)
itp_vals |> tc #hide
````

````@example diffusion_equation_on_an_annulus
fig, ax, sc = contourf(x, y, reshape(itp_vals, length(x), length(y)), colormap=:matter, levels=-10:2:40)
fig
````

## Just the code
An uncommented version of this example is given below.
You can view the source code for this file [here](https://github.com/SciML/FiniteVolumeMethod.jl/tree/main/docs/src/literate_tutorials/diffusion_equation_on_an_annulus.jl).

```julia
using DelaunayTriangulation, FiniteVolumeMethod, CairoMakie
R₁, R₂ = 0.2, 1.0
inner = CircularArc((R₁, 0.0), (R₁, 0.0), (0.0, 0.0), positive=false)
outer = CircularArc((R₂, 0.0), (R₂, 0.0), (0.0, 0.0))
boundary_nodes = [[[outer]], [[inner]]]
points = NTuple{2,Float64}[]
tri = triangulate(points; boundary_nodes)
A = get_area(tri)
refine!(tri; max_area=1e-4A)
triplot(tri)

mesh = FVMGeometry(tri)

fig = Figure()
ax = Axis(fig[1, 1])
outer = [get_point(tri, i) for i in get_neighbours(tri, -1)]
inner = [get_point(tri, i) for i in get_neighbours(tri, -2)]
triplot!(ax, tri)
scatter!(ax, outer, color=:red)
scatter!(ax, inner, color=:blue)
fig

outer_bc = (x, y, t, u, p) -> zero(u)
inner_bc = (x, y, t, u, p) -> oftype(u, 50(1 - exp(-t / 2)))
types = (Neumann, Dirichlet)
BCs = BoundaryConditions(mesh, (outer_bc, inner_bc), types)

initial_condition_f = (x, y) -> begin
    10 * exp(-25 * ((x + 0.5) * (x + 0.5) + (y + 0.5) * (y + 0.5))) - 5 * exp(-50 * ((x + 0.3) * (x + 0.3) + (y + 0.5) * (y + 0.5))) - 10 * exp(-45 * ((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)))
end
diffusion_function = (x, y, t, u, p) -> one(u)
initial_condition = [initial_condition_f(x, y) for (x, y) in DelaunayTriangulation.each_point(tri)]
final_time = 2.0
prob = FVMProblem(mesh, BCs;
    diffusion_function,
    final_time,
    initial_condition)

using OrdinaryDiffEq, LinearSolve
sol = solve(prob, TRBDF2(linsolve=KLUFactorization()), saveat=0.2)

fig = Figure(fontsize=38)
for (i, j) in zip(1:3, (1, 6, 11))
    local ax
    ax = Axis(fig[1, i], width=600, height=600,
        xlabel="x", ylabel="y",
        title="t = $(sol.t[j])",
        titlealign=:left)
    tricontourf!(ax, tri, sol.u[j], levels=-10:2:40, colormap=:matter)
    tightlimits!(ax)
end
resize_to_layout!(fig)
fig

add_ghost_triangles!(tri)

x = LinRange(-R₂, R₂, 400)
y = LinRange(-R₂, R₂, 400)
interp_vals = zeros(length(x), length(y))
u = sol.u[6]
last_triangle = Ref((1, 1, 1))
for (j, _y) in enumerate(y)
    for (i, _x) in enumerate(x)
        T = jump_and_march(tri, (_x, _y), try_points=last_triangle[])
        last_triangle[] = triangle_vertices(T) # used to accelerate jump_and_march, since the points we're looking for are close to each other
        if DelaunayTriangulation.is_ghost_triangle(T) # don't extrapolate
            interp_vals[i, j] = NaN
        else
            interp_vals[i, j] = pl_interpolate(prob, T, sol.u[6], _x, _y)
        end
    end
end
fig, ax, sc = contourf(x, y, interp_vals, levels=-10:2:40, colormap=:matter)
fig

using NaturalNeighbours
_x = vec([x for x in x, y in y]) # NaturalNeighbours.jl needs vector data
_y = vec([y for x in x, y in y])
itp = interpolate(tri, u, derivatives=true)

itp_vals = itp(_x, _y; method=Farin())

fig, ax, sc = contourf(x, y, reshape(itp_vals, length(x), length(y)), colormap=:matter, levels=-10:2:40)
fig

itp_vals = itp(_x, _y; method=Farin(), project=false)

fig, ax, sc = contourf(x, y, reshape(itp_vals, length(x), length(y)), colormap=:matter, levels=-10:2:40)
fig
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

