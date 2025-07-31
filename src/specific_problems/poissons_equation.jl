@doc raw"""
    PoissonsEquation

A struct for defining a problem representing a (generalised) Poisson's equation:
```math
\div[D(\vb x)\grad u] = f(\vb x)
```
inside a domain $\Omega$. See also [`LaplacesEquation`](@ref), a special case of this 
problem with $f(\vb x) = 0$.

You can solve this problem using [`solve`](@ref solve(::AbstractFVMTemplate, args...; kwargs...)).

# Constructor 

    PoissonsEquation(mesh::FVMGeometry,
        BCs::BoundaryConditions,
        ICs::InternalConditions=InternalConditions();
        diffusion_function=(x,y,p)->1.0,
        diffusion_parameters=nothing,
        source_function, 
        source_parameters=nothing,
        kwargs...)

## Arguments
- `mesh::FVMGeometry`: The [`FVMGeometry`](@ref).
- `BCs::BoundaryConditions`: The [`BoundaryConditions`](@ref). For these boundary conditions, all functions should still be of the form `(x, y, t, u, p) -> Number`, but the `t` and `u` arguments should be unused as they will be replaced with `nothing`.
- `ICs::InternalConditions=InternalConditions()`: The [`InternalConditions`](@ref). For these internal conditions, all functions should still be of the form `(x, y, t, u, p) -> Number`, but the `t` and `u` arguments should be unused as they will be replaced with `nothing`.

## Keyword Arguments
- `diffusion_function=(x,y,p)->1.0`: The diffusion function. Should be of the form `(x, y, p) -> Number`, where `p = diffusion_parameters` below.
- `diffusion_parameters=nothing`: The argument `p` in `diffusion_function`.
- `source_function`: The source function. Should be of the form `(x, y, p) -> Number`, where `p = source_parameters` below.
- `source_parameters=nothing`: The argument `p` in `source_function`.
- `kwargs...`: Any other keyword arguments are passed to the `LinearProblem` (from LinearSolve.jl) that represents the problem.

# Fields
The struct has extra fields in addition to the arguments above:
- `A`: This is a sparse matrix `A` so that `Au = b`.    
- `b`: The `b` above.
- `problem`: The `LinearProblem` that represents the problem. This is the problem that is solved when you call [`solve`](@ref solve(::AbstractFVMTemplate, args...; kwargs...)) on the struct.
"""
struct PoissonsEquation{M, C, D, DP, S, SP, A, B, ODE} <: AbstractFVMTemplate
    mesh::M
    conditions::C
    diffusion_function::D
    diffusion_parameters::DP
    source_function::S
    source_parameters::SP
    A::A
    b::B
    problem::ODE
end
function Base.show(io::IO, ::MIME"text/plain", prob::PoissonsEquation)
    nv = DelaunayTriangulation.num_solid_vertices(prob.mesh.triangulation)
    print(io, "PoissonsEquation with $(nv) nodes")
end

function PoissonsEquation(mesh::FVMGeometry,
        BCs::BoundaryConditions,
        ICs::InternalConditions = InternalConditions();
        diffusion_function = (x, y, p) -> 1.0,
        diffusion_parameters = nothing,
        source_function,
        source_parameters = nothing,
        kwargs...)
    conditions = Conditions(mesh, BCs, ICs)
    has_dudt_nodes(conditions) &&
        throw(ArgumentError("PoissonsEquation does not support Dudt nodes."))
    n = DelaunayTriangulation.num_points(mesh.triangulation)
    A = zeros(n, n)
    b = create_rhs_b(mesh, conditions, source_function, source_parameters)
    triangle_contributions!(A, mesh, conditions, diffusion_function, diffusion_parameters)
    boundary_edge_contributions!(
        A, b, mesh, conditions, diffusion_function, diffusion_parameters)
    apply_steady_dirichlet_conditions!(A, b, mesh, conditions)
    fix_missing_vertices!(A, b, mesh)
    Asp = sparse(A)
    prob = LinearProblem(Asp, b; kwargs...)
    return PoissonsEquation(mesh, conditions,
        diffusion_function, diffusion_parameters,
        source_function, source_parameters,
        Asp, b, prob)
end
