```@meta
EditURL = "https://github.com/DanielVandH/FiniteVolumeMethod.jl/tree/main/docs/src/literate_tutorials/heat_convection_on_a_square_plate_with_inhomogeneous_neumann_boundary_conditions_and_a_robin_boundary_condition.jl"
```

# Heat Convection on a Square Plate with Inhomogeneous Neumann conditions and a Robin boundary condition
In this tutorial, we consider the following problem:
```math
\begin{equation*}
\begin{aligned}
\frac{1}{\alpha}\pdv{T}{t} &= \pdv[2]{T}{x} + \pdv[2]{T}{y} & 0 < x < L,\, 0 < y < L, \\[6pt]
\pdv{T}{x} &= 0 & x=0,\,x=L, \\[6pt]
\pdv{T}{y} &= -\frac{q}{k} & y=0, \\[6pt]
k\pdv{T}{y} + hT &= hT_{\infty} & y=L,\\[6pt]
T(x, y, 0) &= T_0.
\end{aligned}
\end{equation*}
```
This is the first example we have considered thus that involves a Robin
boundary condition, and also the first that involves an inhomogeneous
Neumann boundary condition. We need to rewrite the Robin boundary condition
as a Neumann boundary condition, which we can do by rearranging:
```math
\pdv{T}{y} = \frac{h}{k}\left(T_{\infty} - T\right), \quad y = L.
```
Note also that we do require that boundary conditions are given in the form
$\vb q \vdot \vu n = a$, where $\vu n$ is the unit normal vector for the
associated boundary and $a$ is some function. To check if our boundary conditions are in this form, we note that:
```math
\vb q = -\alpha\grad T.
```
Moreover, $\vu n$ is the unit normal vector for the boundary, which is given by
```math
\vu n = \begin{cases} -\vu i & x = 0, \\ \vu i & x = L, \\ -\vu j & y = 0, \\ \vu j & y = L. \end{cases}
```
Thus,
```math
\vb q \vdot \vu n = \begin{cases} \alpha\pdv{T}{x} & x = 0, \\[6pt] -\alpha\pdv{T}{x} & x = L, \\[6pt] \alpha\pdv{T}{y} & y = 0, \\ -\alpha\pdv{T}{y} & y = L. \end{cases}
```
So, using our given boundary conditions, we write them in the form $\vb q \vdot \vu n = a$ as follows:
```math
\vb q \vdot \vu n = \begin{cases} 0 & x=0,\\[6pt] 0 & x = L, \\[6pt] -\frac{\alpha q}{k} & y=0, \\[6pt] -\frac{\alpha h}{k}\left(T_{\infty} - T\right) & y = L. \end{cases}
```
Now that we've expressed the boundary conditions in the required form, let us define the problem. The boundary
condition for each side needs to be expressed using a function `a(x, y, t, T, p) -> Number`, as we do below.

````@example heat_convection_on_a_square_plate_with_inhomogeneous_neumann_boundary_conditions_and_a_robin_boundary_condition
using DelaunayTriangulation, FiniteVolumeMethod
L = 1.0
k = 237.0
T₀ = 10.0
T∞ = 10.0
α = 80e-6
q = 10.0
h = 25.0
tri = triangulate_rectangle(0, L, 0, L, 200, 200; single_boundary=false)
mesh = FVMGeometry(tri)
````

````@example heat_convection_on_a_square_plate_with_inhomogeneous_neumann_boundary_conditions_and_a_robin_boundary_condition
bot_wall = (x, y, t, T, p) -> -p.α * p.q / p.k
right_wall = (x, y, t, T, p) -> zero(T)
top_wall = (x, y, t, T, p) -> -p.α * p.h / p.k * (p.T∞ - T)
left_wall = (x, y, t, T, p) -> zero(T)
bc_fncs = (bot_wall, right_wall, top_wall, left_wall) # the order is important
types = (Neumann, Neumann, Neumann, Neumann)
bot_parameters = (α=α, q=q, k=k)
right_parameters = nothing
top_parameters = (α=α, h=h, k=k, T∞=T∞)
left_parameters = nothing
parameters = (bot_parameters, right_parameters, top_parameters, left_parameters)
BCs = BoundaryConditions(mesh, bc_fncs, types; parameters)
````

````@example heat_convection_on_a_square_plate_with_inhomogeneous_neumann_boundary_conditions_and_a_robin_boundary_condition
flux_function = (x, y, t, α, β, γ, p) -> begin
    ∇u = (α, β)
    return -p.α .* ∇u
end
flux_parameters = (α=α,)
final_time = 2000.0
f = (x, y) -> T₀
initial_condition = [f(x, y) for (x, y) in each_point(tri)]
prob = FVMProblem(mesh, BCs;
    flux_function,
    flux_parameters,
    initial_condition,
    final_time)
````

Now let us solve this problem.

````@example heat_convection_on_a_square_plate_with_inhomogeneous_neumann_boundary_conditions_and_a_robin_boundary_condition
using OrdinaryDiffEq, LinearSolve
sol = solve(prob, TRBDF2(linsolve=KLUFactorization()); saveat=10.0)
````

Now we visualise.

````@example heat_convection_on_a_square_plate_with_inhomogeneous_neumann_boundary_conditions_and_a_robin_boundary_condition
using CairoMakie
using ReferenceTests #src
times = 0:500:2000
t_idx = [findlast(≤(t), sol.t) for t in times]
fig = Figure(fontsize=38)
for (i, j) in enumerate(t_idx)
    ax = Axis(fig[1, i], width=600, height=600,
        xlabel="x", ylabel="y",
        title="t = $(sol.t[j])",
        titlealign=:left)
    tricontourf!(ax, tri, sol.u[j], levels=10:0.1:12.0, colormap=:matter, extendlow=:auto)
    tightlimits!(ax)
end
resize_to_layout!(fig)
fig

using Roots #src
````

## Just the code
An uncommented version of this example is given below.
You can view the source code for this file [here](https://github.com/DanielVandH/FiniteVolumeMethod.jl/tree/new-docs/docs/src/literate_tutorials/heat_convection_on_a_square_plate_with_inhomogeneous_neumann_boundary_conditions_and_a_robin_boundary_condition.jl).

```julia
using DelaunayTriangulation, FiniteVolumeMethod
L = 1.0
k = 237.0
T₀ = 10.0
T∞ = 10.0
α = 80e-6
q = 10.0
h = 25.0
tri = triangulate_rectangle(0, L, 0, L, 200, 200; single_boundary=false)
mesh = FVMGeometry(tri)

bot_wall = (x, y, t, T, p) -> -p.α * p.q / p.k
right_wall = (x, y, t, T, p) -> zero(T)
top_wall = (x, y, t, T, p) -> -p.α * p.h / p.k * (p.T∞ - T)
left_wall = (x, y, t, T, p) -> zero(T)
bc_fncs = (bot_wall, right_wall, top_wall, left_wall) # the order is important
types = (Neumann, Neumann, Neumann, Neumann)
bot_parameters = (α=α, q=q, k=k)
right_parameters = nothing
top_parameters = (α=α, h=h, k=k, T∞=T∞)
left_parameters = nothing
parameters = (bot_parameters, right_parameters, top_parameters, left_parameters)
BCs = BoundaryConditions(mesh, bc_fncs, types; parameters)

flux_function = (x, y, t, α, β, γ, p) -> begin
    ∇u = (α, β)
    return -p.α .* ∇u
end
flux_parameters = (α=α,)
final_time = 2000.0
f = (x, y) -> T₀
initial_condition = [f(x, y) for (x, y) in each_point(tri)]
prob = FVMProblem(mesh, BCs;
    flux_function,
    flux_parameters,
    initial_condition,
    final_time)

using OrdinaryDiffEq, LinearSolve
sol = solve(prob, TRBDF2(linsolve=KLUFactorization()); saveat=10.0)

using CairoMakie
using ReferenceTests #src
times = 0:500:2000
t_idx = [findlast(≤(t), sol.t) for t in times]
fig = Figure(fontsize=38)
for (i, j) in enumerate(t_idx)
    ax = Axis(fig[1, i], width=600, height=600,
        xlabel="x", ylabel="y",
        title="t = $(sol.t[j])",
        titlealign=:left)
    tricontourf!(ax, tri, sol.u[j], levels=10:0.1:12.0, colormap=:matter, extendlow=:auto)
    tightlimits!(ax)
end
resize_to_layout!(fig)
fig

using Roots #src
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

