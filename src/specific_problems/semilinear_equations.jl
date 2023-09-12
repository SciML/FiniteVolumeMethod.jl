@doc raw"""
    SemilinearEquation

A struct for defining a problem representing a semilinear equation:
```math
\pdv{u}{t} = \div\left[D(\vb x)\grad u\right] + f(\vb x, t, u)
```
inside a domain $\Omega$. See also [`SemilinearSystem`](@ref),
which is the system form of this scalar problem.
"""
struct SemilinearEquation <: AbstractFVMTemplate end