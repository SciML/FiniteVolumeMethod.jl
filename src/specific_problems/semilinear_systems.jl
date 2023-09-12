@doc raw"""
    SemilinearSystem

A struct for defining a problem representing a semilinear system:
```math
\pdv{\vb u}{t} = \div\left[\vb D(\vb x)\grad\vb u\right] + \vb F(\vb x, t, \vb u)
```
inside a domain $\Omega$. This is the system form of the corresponding 
scalar problem [`SemilinearSystem`](@ref).
"""
struct SemilinearSystem <: AbstractFVMTemplate end