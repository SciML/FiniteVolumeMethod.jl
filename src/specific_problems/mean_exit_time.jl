@doc raw"""
    MeanExitTimeProblem

A struct for defining a problem representing a mean exit time problem:
```math
\div\left[D(\vb x)\grad T\right] =-1
```
inside a domain $\Omega$. This problem is a special case of [`PoissonsEquation`](@ref),
but is defined separately since it is common enough to warrant its own definition;
`MeanExitTimeProblem` is constructed using [`PoissonsEquation`](@ref).

You can solve this problem using [`solve`](@ref solve(::AbstractFVMTemplate, args...; kwargs...)).

# Constructor 

    MeanExitTimeProblem(mesh::FVMGeometry,
        BCs::BoundaryConditions,
        ICs::InternalConditions=InternalConditions();
        diffusion_function,
        diffusion_parameters=nothing,
        kwargs...)

## Arguments 
- `mesh::FVMGeometry`: The [`FVMGeometry`](@ref).
- `BCs::BoundaryConditions`: The [`BoundaryConditions`](@ref).
- `ICs::InternalConditions=InternalConditions()`: The [`InternalConditions`](@ref).

The functions for `BCs` and `ICs` are not used. Whenever a [`Neumann`](@ref) condition is encountered, 
or a [`Dirichlet`](@ref) condition, it is assumed that the conditon is homogeneous. If any of the 
conditions are [`Dudt`](@ref) or [`Constrained`](@ref) types, then an error is thrown.

## Keyword Arguments
- `diffusion_function`: The diffusion function. Should be of the form `(x, y, p) -> Number`, where `p = diffusion_parameters` below.
- `diffusion_parameters=nothing`: The argument `p` in `diffusion_function`.
- `kwargs...`: Any other keyword arguments are passed to the `LinearProblem` (from LinearSolve.jl) that represents the problem.

# Fields
The struct has extra fields in addition to the arguments above:
- `A`: This is a sparse matrix `A` so that `AT = b`.
- `b`: The `b` above.
- `problem`: The `LinearProblem` that represents the problem. This is the problem that is solved when you call [`solve`](@ref solve(::AbstractFVMTemplate, args...; kwargs...)) on the struct.
"""
struct MeanExitTimeProblem{M,C,D,DP,A,B,LP} <: AbstractFVMTemplate
    mesh::M
    conditions::C
    diffusion_function::D
    diffusion_parameters::DP
    A::A
    b::B
    problem::LP
end
function Base.show(io::IO, ::MIME"text/plain", prob::MeanExitTimeProblem)
    nv = DelaunayTriangulation.num_solid_vertices(prob.mesh.triangulation)
    print(io, "MeanExitTimeProblem with $(nv) nodes")
end

function MeanExitTimeProblem(mesh::FVMGeometry,
    BCs::BoundaryConditions,
    ICs::InternalConditions=InternalConditions();
    diffusion_function,
    diffusion_parameters=nothing,
    kwargs...)
    conditions = Conditions(mesh, BCs, ICs)
    has_dudt_nodes(conditions) && throw(ArgumentError("MeanExitTimeProblem does not support Dudt nodes."))
    has_constrained_edges(conditions) && throw(ArgumentError("MeanExitTimeProblem does not support Constrained edges."))
    n = DelaunayTriangulation.num_solid_vertices(mesh.triangulation)
    A = zeros(n, n)
    triangle_contributions!(A, mesh, conditions, diffusion_function, diffusion_parameters)
    b = create_met_b!(A, mesh, conditions)
    Asp = sparse(A)
    prob = LinearProblem(Asp, b; kwargs...)
    return MeanExitTimeProblem(mesh, conditions,
        diffusion_function, diffusion_parameters,
        Asp, b, prob)
end

function create_met_b!(A, mesh, conditions)
    return create_rhs_b!(A, mesh, conditions, (x, y, p) -> -1.0, nothing)
end