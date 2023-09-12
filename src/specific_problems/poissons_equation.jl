@doc raw"""
    PoissonsEquation

A struct for defining a problem representing a Poisson's equation:
```math
\grad^2 u = f(\vb x)
```
inside a domain $\Omega$. See also [`LaplacesEquation`](@ref), a special case of this 
problem with $f(\vb x) = 0$.
"""
struct PoissonsEquation <: AbstractFVMTemplate end
