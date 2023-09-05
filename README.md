# FiniteVolumeMethod

[![DOI](https://zenodo.org/badge/561533716.svg)](https://zenodo.org/badge/latestdoi/561533716)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://DanielVandH.github.io/FiniteVolumeMethod.jl/dev)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://DanielVandH.github.io/FiniteVolumeMethod.jl/stable)
[![Coverage](https://codecov.io/gh/DanielVandH/FiniteVolumeMethod.jl/branch/main/graph/badge.svg?token=XPM5KN89R6)](https://codecov.io/gh/DanielVandH/FiniteVolumeMethod.jl)

This is a Julia package for solving partial differential equations (PDEs) of the form 

$$
\dfrac{\partial u(\boldsymbol x, t)}{\partial t} + \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{q}(\boldsymbol x, t, u) = S(\boldsymbol x, t, u), \quad (x, y)^{\mkern-1.5mu\mathsf{T}} \in \Omega \subset \mathbb R^2,t>0,
$$

in two dimensions using the finite volume method, with support also provided for steady-state problems and for systems of PDEs of the above form. 

If this package doesn't suit what you need, you may like to review some of the other PDE packages shown [here](https://github.com/JuliaPDE/SurveyofPDEPackages).

 As a very quick demonstration, here is how we could solve a diffusion equation with Dirichlet boundary conditions on a square domain; please see the docs for more information.

```julia
using FiniteVolumeMethod, DelaunayTriangulation, CairoMakie, DifferentialEquations
a, b, c, d = 0.0, 2.0, 0.0, 2.0
nx, ny = 50, 50
tri = triangulate_rectangle(a, b, c, d, nx, ny, single_boundary=true)
mesh = FVMGeometry(tri)
bc = (x, y, t, u, p) -> zero(u)
BCs = BoundaryConditions(mesh, bc, Dirichlet)
f = (x, y) -> y ≤ 1.0 ? 50.0 : 0.0
initial_condition = [f(x, y) for (x, y) in each_point(tri)]
D = (x, y, t, u, p) -> 1 / 9
final_time = 0.5
prob = FVMProblem(mesh, BCs; diffusion_function=D, initial_condition, final_time)
sol = solve(prob, saveat=0.05)
u = Observable(sol.u[1])
fig, ax, sc = tricontourf(tri, u, levels=0:5:50, colormap=:matter)
tightlimits!(ax)
record(fig, "anim.mp4", eachindex(sol)) do i
    u[] = sol.u[i]
end
```

![Animation of a solution](https://github.com/DanielVandH/FiniteVolumeMethod.jl/blob/imp_clean/anim.mp4?raw=true)