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