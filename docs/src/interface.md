```@meta
CurrentModule = FiniteVolumeMethod
```

# Interface 

In this section, we describe the basic interface for defining and solving PDEs using this package. This interface will also be made clearer in the tutorials. The basic summary of the discussion below is as follows:

1. Use `FVMGeometry` to define the problem's mesh.
2. Provide boundary conditions using `BoundaryConditions`.
3. (Optional) Provide internal conditions using `InternalConditions`.
4. Convert the problem into an `FVMProblem`.
5. If you want to make the problem steady, use `SteadyFVMProblem` on the `FVMProblem`.
6. If you want a system of equations, construct an `FVMSystem` from multiple `FVMProblem`s; if you want this problem to be steady, skip step 5 and only now apply `SteadyFVMProblem`.
7. Solve the problem using `solve`.
8. For a discussion of custom constraints, see the tutorials.
9. For interpolation, we provide `pl_interpolate` (but you might prefer [NaturalNeighbours.jl](https://github.com/DanielVandH/NaturalNeighbours.jl) - see [this tutorial for an example](tutorials/piecewise_linear_and_natural_neighbour_interpolation_for_an_advection_diffusion_equation.md)).

## `FVMGeometry`: Defining the mesh 

The finite volume method (FVM) requires an underlying triangular mesh, as outlined in the [mathematical details section](math.md). This triangular mesh is to be defined from [DelaunayTriangulation.jl](https://github.com/DanielVandH/DelaunayTriangulation.jl). The `FVMGeometry` type wraps the resulting `Triangulation` and computes information about the geometry required for solving the PDEs. The docstring for `FVMGeometry` is below; the fields of `FVMGeometry` are not public API, only this wrapper is.

```@docs
FVMGeometry
```

## `BoundaryConditions`: Defining boundary conditions

Once a mesh is defined, you need to associate each part of the boundary with a set of boundary nodes. Since you have a `Triangulation`, the boundary of the mesh already meets the necessary assumptions made by this package about the boundary; these assumptions are simply that they match the specification of a boundary [here in DelaunayTriangulation.jl's docs](https://danielvandh.github.io/DelaunayTriangulation.jl/dev/boundary_handling/#Boundary-Specification) (for example, the boundary points connect, the boundary is positively oriented, etc.).

You can specify boundary condtiions using `BoundaryConditions`, whose docstring is below; the fields of `BoundaryConditions` are not public API, only this wrapper is.

```@docs
BoundaryConditions
```

There are four types of boundary conditions: `Neumann`, `Dudt`, `Dirichlet`, and `Constrained`. These types are defined below.

```@docs
ConditionType 
Neumann 
Dudt
Dirichlet 
Constrained
```

## `InternalConditions`: Defining internal conditions

If you like, you can also put some constraints for nodes away from the boundary. In this case, only `Dudt` and `Dirichlet` conditions can be imposed; for `Neumann` or `Constrained` conditions, you need to consider differential-algebraic problems as considered in the tutorials. The docstring for `InternalConditions` is below; the fields of `InternalConditions` are not public API, only this wrapper is.

```@docs
InternalConditions
```

## `FVMProblem`: Defining the PDE

Once you have defined the mesh, the boundary conditions, and possibly the internal conditions, you can now construct the PDE itself. This is done using `FVMProblem`, whose docstring is below; the fields of `FVMProblem` are public API.

```@docs
FVMProblem 
```

For this problem, you can provide either a `diffusion_function` or a `flux_function`. In the former case, the `flux_function` is constructed from `diffusion_function` using `construct_flux_function`, whose docstring is shown below; `construct_flux_function` is not public API.

```@docs
construct_flux_function
```

Additionally, `FVMProblem` merges the provided boundary conditions and internal conditions into a `Conditions` type, defined below; the documented fields of `Conditions` are public API.

```@docs
Conditions
```

## `SteadyFVMProblem`: Making the problem a steady-state problem

To make an `FVMProblem` a steady-state problem, meaning that you are solving

```math
\div \vb q(\vb x, t, u) = S(\vb x, t, u)
```

rather than

```math
\pdv{u(\vb x, t)}{t} + \div \vb q(\vb x, t, u) = S(\vb x, t, u),
```

than you need to wrap the `FVMProblem` inside a `SteadyFVMProblem`, defined below; the fields of `SteadyFVMProblem` are not public API, only this wrapper is.

```@docs
SteadyFVMProblem
```

## `FVMSystem`: Defining a system of PDEs

We also allow for systems of PDEs to be defined, where this system should take the form

```math
\begin{equation*}
\begin{aligned}
\pdv{u_1(\vb x, t)}{t} + \div \vb q_1(\vb x, t, u_1, \ldots, u_n) &= S_1(\vb x, t, u_1, \ldots, u_n), \\
\pdv{u_2(\vb x, t)}{t} + \div \vb q_2(\vb x, t, u_1, \ldots, u_n) &= S_2(\vb x, t, u_1, \ldots, u_n), \\
&\vdots \\
\pdv{u_n(\vb x, t)}{t} + \div \vb q_n(\vb x, t, u_1, \ldots, u_n) &= S_n(\vb x, t, u_1, \ldots, u_n).
\end{aligned}
\end{equation*}
```

To define this system, you need to provide an `FVMProblem` for each equation, and then construct an `FVMSystem` from these problems. The docstring for `FVMSystem` is below; the fields of `FVMSystem` are not public API, only this wrapper is.

```@docs
FVMSystem
```

If you want to make a steady-state problem for an `FVMSystem`, you should apply `SteadyFVMProblem` to `FVMSystem` rather than to each `FVMProblem` individually.

## `solve`: Solving the PDE

You can use `solve` from the SciMLBase ecosystem to solve these PDEs. This allows you to use [any of the available algorithms from DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/) for solving these problems. For non-steady problems, the relevant function is (which is public API)

```@docs
solve(::Union{FVMProblem,FVMSystem}, ::Any; kwargs...)
```

For steady-state problems, the algorithms to use are those from [NonlinearSolve.jl](https://docs.sciml.ai/NonlinearSolve/stable/). The relevant function is still `solve` and is public API:

```@docs
solve(::SteadyFVMProblem, ::Any; kwargs...)
```

## Custom constraints 

You can also provide custom constraints. Rather than outlining this here, it is best explained in the tutorials. We note that one useful function for this is `compute_flux`, which allows you to compute the flux across a given edge. The docstring for `compute_flux` is below, and this function is public API.

```@docs
compute_flux
```

## Piecewise linear interpolation

You can evaluate the piecewise linear interpolation corresponding to a solution using `pl_interpolant`, defined below.

```@docs
pl_interpolate
```

Better interpolants are available from [NaturalNeighbours.jl](https://github.com/DanielVandH/NaturalNeighbours.jl) - see the [this tutorial](tutorials/piecewise_linear_and_natural_neighbour_interpolation_for_an_advection_diffusion_equation.md) for some examples.