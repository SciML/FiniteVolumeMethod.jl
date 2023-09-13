@doc raw"""
    LinearReactionDiffusionEquation

A struct for defining a problem representing a linear reaction-diffusion equation:
```math
\pdv{u}{t} = \div\left[D(\vb x)\grad u\right] + f(\vb x)u
```
inside a domain $\Omega$. You can solve this problem using [`solve`](@ref solve(::AbstractFVMTemplate, args...; kwargs...)).

!!! warning 

    The solution to this problem will have an extra component added to it. The original solution will be inside 
    `sol[begin:end-1, :]`, where `sol` is the solution returned by [`solve`](@ref solve(::AbstractFVMTemplate, args...; kwargs...)).

# Constructor 

    LinearReactionDiffusionEquation(mesh::FVMGeometry,
        BCs::BoundaryConditions,
        ICs::InternalConditions=InternalConditions();
        diffusion_function,
        diffusion_parameters=nothing,
        source_function,
        source_parameters=nothing,
        initial_condition,
        initial_time=0.0,
        final_time,
        kwargs...)

## Arguments
- `mesh::FVMGeometry`: The [`FVMGeometry`](@ref).
- `BCs::BoundaryConditions`: The [`BoundaryConditions`](@ref). For these boundary conditions, all functions should still be of the form `(x, y, t, u, p) -> Number`, but the `t` and `u` arguments should be unused as they will be replaced with `nothing`.
- `ICs::InternalConditions=InternalConditions()`: The [`InternalConditions`](@ref). For these internal conditions, all functions should still be of the form `(x, y, t, u, p) -> Number`, but the `t` and `u` arguments should be unused as they will be replaced with `nothing`.

## Keyword Arguments
- `diffusion_function`: The diffusion function. Should be of the form `(x, y, p) -> Number`, where `p = diffusion_parameters` below.
- `diffusion_parameters=nothing`: The argument `p` in `diffusion_function`.
- `source_function`: The source function. Should be of the form `(x, y, p) -> Number`, where `p = source_parameters` below.
- `source_parameters=nothing`: The argument `p` in `source_function`.
- `initial_condition`: The initial condition.
- `initial_time=0.0`: The initial time.
- `final_time`: The final time.
- `kwargs...`: Any other keyword arguments are passed to the `ODEProblem` (from DifferentialEquations.jl) that represents the problem.

# Fields
The struct has extra fields in addition to the arguments above:
- `A`: This is a sparse matrix `A` so that `du/dt = Au + b`.
- `b`: The `b` above.
- `Aop`: The `MatrixOperator` that represents the system so that `du/dt = Aop*u` (with `u` padded with an extra component since `A` is now inside `Aop`).
- `problem`: The `ODEProblem` that represents the problem. This is the problem that is solved when you call [`solve`](@ref solve(::AbstractFVMTemplate, args...; kwargs...)) on the struct.
"""
struct LinearReactionDiffusionEquation{M,C,D,DP,S,SP,IC,FT,A,B,OP,ODE} <: AbstractFVMTemplate
    mesh::M
    conditions::C
    diffusion_function::D
    diffusion_parameters::DP
    source_function::S
    source_parameters::SP
    initial_condition::IC
    initial_time::FT
    final_time::FT
    A::A
    b::B
    Aop::OP
    problem::ODE
end
function Base.show(io::IO, ::MIME"text/plain", prob::LinearReactionDiffusionEquation)
    nv = DelaunayTriangulation.num_solid_vertices(prob.mesh.triangulation)
    t0 = prob.initial_time
    tf = prob.final_time
    print(io, "LinearReactionDiffusionEquation with $(nv) nodes and time span ($t0, $tf)")
end

function LinearReactionDiffusionEquation(mesh::FVMGeometry,
    BCs::BoundaryConditions,
    ICs::InternalConditions=InternalConditions();
    diffusion_function,
    diffusion_parameters=nothing,
    source_function,
    source_parameters=nothing,
    initial_condition,
    initial_time=0.0,
    final_time,
    kwargs...)
    conditions = Conditions(mesh, BCs, ICs)
    n = DelaunayTriangulation.num_solid_vertices(mesh.triangulation)
    Afull = zeros(n + 1, n + 1)
    A = @views Afull[begin:end-1, begin:end-1]
    b = @views Afull[begin:end-1, end]
    _ic = vcat(initial_condition, 1)
    triangle_contributions!(A, mesh, conditions, diffusion_function, diffusion_parameters)
    boundary_edge_contributions!(A, b, mesh, conditions, diffusion_function, diffusion_parameters)
    linear_source_contributions!(A, mesh, conditions, source_function, source_parameters)
    apply_dudt_conditions!(b, mesh, conditions)
    apply_dirichlet_conditions!(_ic, mesh, conditions)
    Af = sparse(Afull)
    Aop = MatrixOperator(Af)
    prob = ODEProblem(Aop, _ic, (initial_time, final_time); kwargs...)
    return LinearReactionDiffusionEquation(mesh, conditions, diffusion_function, diffusion_parameters,
        source_function, source_parameters, initial_condition, initial_time, final_time, sparse(A), b, Aop, prob)
end