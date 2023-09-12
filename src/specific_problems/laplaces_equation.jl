@doc raw"""
    laplaces_equation_with_internal_dirichlet_conditions

A struct for defining a problem representing a Poisson's equation:
```math
\grad^2 u = 0
```
inside a domain $\Omega$. See also [`PoissonsEquation`](@ref).
"""
struct LaplacesEquation <: AbstractFVMTemplate end