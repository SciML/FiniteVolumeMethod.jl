@doc raw"""
    PoissonsEquation

A struct for defining a problem representing a (generalised) Poisson's equation:
```math
\div[D(\vb x)\grad u] = f(\vb x)
```
inside a domain $\Omega$. See also [`LaplacesEquation`](@ref), a special case of this 
problem with $f(\vb x) = 0$.
"""
struct PoissonsEquation <: AbstractFVMTemplate end
