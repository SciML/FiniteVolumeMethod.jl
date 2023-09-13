@doc raw"""
    LaplacesEquation

A struct for defining a problem representing a (generalised) Laplace's equation:
```math
\div[D(\vb x)\grad u] = 0
```
inside a domain $\Omega$. See also [`PoissonsEquation`](@ref).
"""
struct LaplacesEquation <: AbstractFVMTemplate end