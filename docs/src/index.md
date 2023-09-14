```@meta
CurrentModule = FiniteVolumeMethod
```

# Introduction 

This is the documentation for FiniteVolumeMethod.jl. [Click here to go back to the GitHub repository](https://github.com/DanielVandH/FiniteVolumeMethod.jl).

This is a Julia package for solving partial differential equations (PDEs) of the form

```math
\pdv{u(\vb x, t)}{t} + \div \vb q(\vb x, t, u) = S(\vb x, t, u), \quad (x, y)^{\mkern-1.5mu\mathsf{T}} \in \Omega \subset \mathbb R^2,\,t>0,
```

using the finite volume method, with additional support for steady-state problems and for systems of PDEs of the above form. We support Neumann, Dirichlet, and boundary conditions on $\mathrm du/\mathrm dt$, as well as internal conditions and custom constraints. We also provide an interface for solving special cases of the above PDE, namely reaction-diffusion equations

```math
\pdv{u(\vb x, t)}{t} = \div\left[D(\vb x, t, u)\grad u(\vb x, t)\right] + S(\vb x, t, u).
```

The [tutorials](tutorials/overview.md) in the sidebar demonstrate the many possibilities of this package. In addition to these two generic forms, we also provide support for specific problems that can be solved in a more efficient manner, namely:

1. `DiffusionEquation`s: $\partial_tu = \div[D(\vb x)\grad u]$.
2. `MeanExitTimeProblem`s: $\div[D(\vb x)\grad T(\vb x)] = -1$.
3. `LinearReactionDiffusionEquation`s: $\partial_tu + \div[D(\vb x)\grad u] + f(\vb x)u$.
4. `PoissonsEquation`: $\div[D(\vb x)\grad u]  f(\vb x)$.
5. `LaplacesEquation`: $\div[D(\vb x)\grad u] = 0$.

See the [Solvers for Specific Problems, and Writing Your Own](wyos/overview.md) section for more information on these templates.