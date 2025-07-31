# Solvers for Specific Problems, and Writing Your Own

The problems solved by this package are quite general, taking the form

```math
\pdv{u}{t} + \div\vb q = S.
```

For some problems, though, this is not the most efficient form to implement.
For example, the diffusion equation

```math
\pdv{u}{t} = D\grad^2 u
```

might be better treated by converting the problem into

```math
\dv{\vb u}{t} = \vb A\vb u + \vb b,
```

which is faster to solve than if we were to treat it as a nonlinear problem
(which is done by default). For this reason, we define some templates
for specific types of problems, namely:

 1. `DiffusionEquation`s: $\partial_tu = \div[D(\vb x)\grad u]$.
 2. `MeanExitTimeProblem`s: $\div[D(\vb x)\grad T(\vb x)] = -1$.
 3. `LinearReactionDiffusionEquation`s: $\partial_tu = \div[D(\vb x)\grad u] + f(\vb x)u$.
 4. `PoissonsEquation`: $\div[D(\vb x)\grad u] = f(\vb x)$.
 5. `LaplacesEquation`: $\div[D(\vb x)\grad u] = 0$.

The docstrings below define the templates for these problems.

```@docs
FiniteVolumeMethod.AbstractFVMTemplate
solve(::FiniteVolumeMethod.AbstractFVMTemplate, args...; kwargs...)
DiffusionEquation
MeanExitTimeProblem
LinearReactionDiffusionEquation
PoissonsEquation
LaplacesEquation
```

Now, again, we note that all these problems can already be implemented using the main interface `FVMProblem`. However, the templates we provide are more efficient, and also provide a good starting point for writing your own solver, meaning your own function
that evaluates the system of ODEs. In the sections that follow, we will demonstrate two things for each of the problems above:

 1. The mathematical details involved in implementing each template.
 2. Examples of using the templates from FiniteVolumeMethod.jl.

With these two steps, you should be able to also know how to write your own solver for any problem you like.

## Relevant Docstrings for Writing Your Own Solver

For writing these solvers, there are some specific functions that might be of use to you.
Here, we provide the docstrings for these functions. These functions are public API.

```@docs
FiniteVolumeMethod.get_dirichlet_fidx
FiniteVolumeMethod.is_dirichlet_node
FiniteVolumeMethod.get_dirichlet_nodes 
FiniteVolumeMethod.has_dirichlet_nodes 
FiniteVolumeMethod.get_dudt_fidx 
FiniteVolumeMethod.is_dudt_node 
FiniteVolumeMethod.get_dudt_nodes 
FiniteVolumeMethod.has_dudt_nodes
FiniteVolumeMethod.get_neumann_fidx 
FiniteVolumeMethod.is_neumann_edge 
FiniteVolumeMethod.has_neumann_edges
FiniteVolumeMethod.get_neumann_edges
FiniteVolumeMethod.get_constrained_fidx 
FiniteVolumeMethod.is_constrained_edge
FiniteVolumeMethod.has_constrained_edges
FiniteVolumeMethod.get_constrained_edges
FiniteVolumeMethod.eval_condition_fnc
FiniteVolumeMethod.has_condition 
FiniteVolumeMethod.get_cv_components 
FiniteVolumeMethod.get_boundary_cv_components
FiniteVolumeMethod.get_triangle_props 
FiniteVolumeMethod.get_volume 
FiniteVolumeMethod.DelaunayTriangulation.get_point(::FVMGeometry, ::Any)
FiniteVolumeMethod.triangle_contributions!
FiniteVolumeMethod.apply_dirichlet_conditions!
FiniteVolumeMethod.apply_dudt_conditions! 
FiniteVolumeMethod.boundary_edge_contributions!
FiniteVolumeMethod.non_neumann_boundary_edge_contributions!
FiniteVolumeMethod.neumann_boundary_edge_contributions!
FiniteVolumeMethod.create_rhs_b
FiniteVolumeMethod.apply_steady_dirichlet_conditions!
FiniteVolumeMethod.two_point_interpolant
FiniteVolumeMethod.get_dirichlet_callback
FiniteVolumeMethod.jacobian_sparsity
FiniteVolumeMethod.fix_missing_vertices!
```
