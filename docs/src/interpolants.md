# Example VI: Using the linear interpolants

Please ensure you have read the List of Examples section before proceeding. This example also reuses some results from the end of Example V.

We now give an example of how one can efficiently evaluate the linear interpolants for a given solution. We illustrate this using the porous medium equation with a linear source example. Letting `prob` be as we computed in that example, we find the solution:

```julia
using OrdinaryDiffEq, LinearSolve

alg = TRBDF2(linsolve=KLUFactorization(), autodiff=true)
sol = solve(prob, alg, saveat=2.5)
```

If we just have a single point $(x, y)$ to evaluate the interpolant at, for a given $t$, we can do the following. First, we define the triple $(x, y, t)$:

```julia
x = 0.37 
y = 0.58
t_idx = 5 # t = sol.t[t_idx]
```

Next, we must find what triangle contains `(x, y)`. This is done by calling into the point location method provided by DelaunayTriangulation.jl, namely `jump_and_march`. We provide a simple interface for this using `FVMProblem`, which we use as follows:
```julia
using DelaunayTriangulation, Test
const DT = DelaunayTriangulation

V = jump_and_march(prob, (x, y))
@test DT.is_inside(DT.point_position_relative_to_triangle(get_point(prob, V...)..., (x, y)))
```
(You can also provide keyword arguments to `jump_and_march`, matching those from DelaunayTriangulation.jl.) Now we can evaluate the interpolant at this point:
```julia
val = eval_interpolant(sol, x, y, t_idx, V)
# or eval_interpolant(sol, x, y, sol.t[t_idx], V)
```
This is our approximation to $u(0.37, 0.58, 0.2)$.

A more typical example would involve evaluating this interpolant over a much larger set of points. A good way to do this is to first find all the triangles that correspond to each point. In what follows, we define a lattice of points, and then we find the triangle for each point. To accelerate the procedure, when initiating the `jump_and_march` function we will tell it to also try starting at the previously found triangle. Note that we also put the grid slightly off the boundary so that all points in a triangle, including those on the boundary.
```julia
nx = 250
ny = 250
grid_x = LinRange(-L + 1e-1, L - 1e-1, nx)
grid_y = LinRange(-L + 1e-1, L - 1e-1, ny)
V_mat = Matrix{NTuple{3, Int64}}(undef, nx, ny)
last_triangle = first(FiniteVolumeMethod.get_elements(prob)) # initiate 
for j in 1:ny 
    for i in 1:nx 
        V_mat[i, j] = jump_and_march(prob, (grid_x[i], grid_y[j]); try_points = last_triangle)
        last_triangle = V_mat[i, j]
    end
end
```

Now let's evaluate the interpolant at each time.
```julia 
u_vals = zeros(nx, ny, length(sol))
for k in eachindex(sol)
    for j in 1:ny
        for i in 1:nx
            V = V_mat[i, j]
            u_vals[i, j, k] = eval_interpolant(sol, grid_x[i], grid_y[j], k, V)
        end
    end
end
```

This setup now makes it easy to use `surface!` from Makie.jl to visualise the solution, thanks to our regular grid.
```julia
using CairoMakie

fig = Figure(resolution=(2744.0f0, 692.0f0))
for k in 1:4
    ax = Axis3(fig[1, k])
    zlims!(ax, 0, 1), xlims!(ax, -L - 1e-1, L + 1e-1), ylims!(ax, -L - 1e-1, L + 1e-1)
    surface!(ax, grid_x, grid_y, u_vals[:, :, k+1], colormap=:matter)
end 
```

```@raw html
<figure>
    <img src='../figures/surface_plots_travelling_wave.png.png', alt='Surface plots'><br>
</figure>
```