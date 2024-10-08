# FiniteVolumeMethod

[![DOI](https://zenodo.org/badge/561533716.svg)](https://zenodo.org/badge/latestdoi/561533716)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://SciML.github.io/FiniteVolumeMethod.jl/dev)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://SciML.github.io/FiniteVolumeMethod.jl/stable)
[![Coverage](https://codecov.io/gh/SciML/FiniteVolumeMethod.jl/branch/main/graph/badge.svg?token=XPM5KN89R6)](https://codecov.io/gh/SciML/FiniteVolumeMethod.jl)

This is a Julia package for solving partial differential equations (PDEs) of the form 

$$
\dfrac{\partial u(\boldsymbol x, t)}{\partial t} + \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{q}(\boldsymbol x, t, u) = S(\boldsymbol x, t, u), \quad (x, y)^{\mkern-1.5mu\mathsf{T}} \in \Omega \subset \mathbb R^2,t>0,
$$

in two dimensions using the finite volume method, with support also provided for steady-state problems and for systems of PDEs of the above form. In addition to this generic form above, we also provide support for specific problems that can be solved in a more efficient manner, namely:

1. `DiffusionEquation`s: $\partial_tu = \boldsymbol\nabla\boldsymbol\cdot[D(\boldsymbol x)\boldsymbol\nabla u]$.
2. `MeanExitTimeProblem`s: $\boldsymbol\nabla\boldsymbol\cdot[D(\boldsymbol x)\boldsymbol\nabla T(\boldsymbol x)] = -1$.
3. `LinearReactionDiffusionEquation`s: $\partial_tu = \boldsymbol\nabla\boldsymbol\cdot[D(\boldsymbol x)\boldsymbol\nabla u] + f(\boldsymbol x)u$.
4. `PoissonsEquation`: $\boldsymbol\nabla\boldsymbol\cdot[D(\boldsymbol x)\boldsymbol\nabla u] = f(\boldsymbol x)$.
5. `LaplacesEquation`: $\boldsymbol\nabla\boldsymbol\cdot[D(\boldsymbol x)\boldsymbol\nabla u] = 0$.

See the documentation for more information.

If this package doesn't suit what you need, you may like to review some of the other PDE packages shown [here](https://github.com/JuliaPDE/SurveyofPDEPackages).

 As a very quick demonstration, here is how we could solve a diffusion equation with Dirichlet boundary conditions on a square domain using the standard `FVMProblem` formulation; please see the docs for more information.

```julia
using FiniteVolumeMethod, DelaunayTriangulation, CairoMakie, OrdinaryDiffEq
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
u = Observable(sol.u[1])
fig, ax, sc = tricontourf(tri, u, levels=0:5:50, colormap=:matter)
tightlimits!(ax)
record(fig, "anim.gif", eachindex(sol)) do i
    u[] = sol.u[i]
end
```

![Animation of a solution](https://github.com/SciML/FiniteVolumeMethod.jl/blob/main/anim.gif)

We could have equivalently used the `DiffusionEquation` template, so that `prob` could have also been defined by 

```julia
prob = DiffusionEquation(mesh, BCs; diffusion_function=D, initial_condition, final_time)
```

and be solved much more efficiently. See the documentation for more information.
