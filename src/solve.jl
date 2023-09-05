function get_multithreading_vectors(prob::Union{FVMProblem,FVMSystem{N}}) where {N}
    u = prob.initial_condition
    nt = Threads.nthreads()
    if prob isa FVMProblem
        duplicated_du = DiffCache(similar(u, length(u), nt))
        point_conditions = collect(keys(prob.conditions.dirichlet_nodes))
    else
        duplicated_du = DiffCache(similar(u, size(u, 1), size(u, 2), nt))
        point_conditions = ntuple(i -> collect(keys(prob.problems[i].conditions.dirichlet_nodes)), N)
    end
    solid_triangles = collect(each_solid_triangle(prob.mesh.triangulation))
    solid_vertices = collect(each_solid_vertex(prob.mesh.triangulation))
    chunked_solid_triangles = chunks(solid_triangles, nt)
    return (
        duplicated_du=duplicated_du,
        point_conditions=point_conditions,
        solid_triangles=solid_triangles,
        solid_vertices=solid_vertices,
        chunked_solid_triangles=chunked_solid_triangles,
        parallel=Val(true),
        prob=prob
    )
end

"""
    jacobian_sparsity(prob::Union{FVMProblem,FVMSystem})

Constructs the sparse matrix which has the same sparsity pattern as the Jacobian for the finite volume equations 
corresponding to the [`FVMProblem`](@ref) or [`FVMSystem`](@ref) given by `prob`.
"""
function jacobian_sparsity(prob::FVMProblem)
    tri = prob.mesh.triangulation
    I = Int64[]   # row indices 
    J = Int64[]   # col indices 
    V = Float64[] # values (all 1)
    n = length(prob.initial_condition)
    sizehint!(I, 6n) # points have, on average, six neighbours in a DelaunayTriangulation
    sizehint!(J, 6n)
    sizehint!(V, 6n)
    for i in each_solid_vertex(tri)
        push!(I, i)
        push!(J, i)
        push!(V, 1.0)
        ngh = get_neighbours(tri, i)
        for j in ngh
            if !DelaunayTriangulation.is_boundary_index(j)
                push!(I, i)
                push!(J, j)
                push!(V, 1.0)
            end
        end
    end
    return sparse(I, J, V)
end
function jacobian_sparsity(prob::FVMSystem{N}) where {N}
    tri = prob.mesh.triangulation
    I = Int64[]   # row indices
    J = Int64[]   # col indices
    V = Float64[] # values (all 1)
    n = length(prob.initial_condition)
    sizehint!(I, 6n) # points have, on average, six neighbours in a DelaunayTriangulation. We don't need to multiply by N here, since length(prob.initial_condition) is actually N * num_solid_vertices(tri) already 
    sizehint!(J, 6n)
    sizehint!(V, 6n)
    for i in each_solid_vertex(tri)
        for j in 1:N
            push!(I, i)
            push!(J, (j - 1) * N + i)
            push!(V, 1.0)
        end
        ngh = get_neighbours(tri, i)
        for j in ngh
            if !DelaunayTriangulation.is_boundary_index(j)
                for k in 1:N
                    push!(I, i)
                    push!(J, (k - 1) * N + j)
                    push!(V, 1.0)
                end
            end
        end
    end
    return sparse(I, J, V)
end

@inline function dirichlet_callback(has_saveat=true)
    cb = DiscreteCallback(
        Returns(true),
        (integrator, t, u) -> update_dirichlet_nodes!(integrator); save_positions=(!has_saveat, !has_saveat)
    )
    return cb
end

function SciMLBase.ODEProblem(prob::Union{FVMProblem,FVMSystem};
    specialization::Type{S}=SciMLBase.AutoSpecialize,
    jac_prototype=jacobian_sparsity(prob),
    parallel::Bool,
    kwargs...) where {S}
    par = Val(parallel)
    initial_time = prob.initial_time
    final_time = prob.final_time
    time_span = (initial_time, final_time)
    initial_condition = prob.initial_condition
    kwarg_dict = Dict(kwargs)
    dirichlet_cb = dirichlet_callback(:saveat ∈ keys(kwarg_dict))
    if :callback ∈ keys(kwarg_dict)
        callback = CallbackSet(kwarg_dict[:callback], dirichlet_cb)
    else
        callback = CallbackSet(dirichlet_cb)
    end
    delete!(kwargs, :callback)
    f = ODEFunction{true,S}(fvm_eqs!; jac_prototype)
    p = par ? get_multithreading_vectors(prob) : prob
    ode_problem = ODEProblem{true,S}(f, initial_condition, time_span, p; callback=callback, kwargs...)
    return ode_problem
end
function SciMLBase.NonlinearProblem(prob::SteadyFVMProblem; kwargs...)
    ode_prob = ODEProblem(prob.problem; kwargs...)
    nl_prob = NonlinearProblem{true}(ode_prob.f, ode_prob.u0, ode_prob.p; kwargs...)
    return nl_prob
end

CommonSolve.init(prob::Union{FVMProblem,FVMSystem}, alg; kwargs...) = CommonSolve.init(ODEProblem(prob, kwargs...), alg; kwargs...)
CommonSolve.solve(prob::SteadyFVMProblem, alg; kwargs...) = CommonSolve.solve(NonlinearProblem(prob; kwargs...), alg; kwargs...)

@doc """
    solve(prob::Union{FVMProblem,FVMSystem}, alg; kwargs...)

Solves the given [`FVMProblem`](@ref) or [`FVMSystem`](@ref) `prob` with the algorithm `alg`, with keyword 
arguments `kwargs` passed to the solver as in DifferentialEquations.jl. The returned type for a [`FVMProblem`](@ref)
is a `sol::ODESolution`, with the `i`th component of the solution referring to the `i`th 
node in the underlying mesh, and accessed like the solutions in DifferentialEquations.jl. If `prob` is a 
[`FVMSystem`](@ref), the `(j, i)`th component of the solution instead refers to the `i`th node 
for the `j`th component of the system.
""" solve(::Union{FVMProblem,FVMSystem}, ::Any; kwargs...)

@doc """
    solve(prob::SteadyFVMProblem, alg; kwargs...)

Solves the given [`SteadyFVMProblem`](@ref) `prob` with the algorithm `alg`, with keyword
arguments `kwargs` passed to the solver as in (Simple)NonlinearSolve.jl. The returned type
is a `NonlinearSolution`, and the `i`th component of the solution if the steady state for the 
`i`th node in the underlying mesh. If the underlying problem is instead a [`FVMSystem`](@ref), 
rather than a [`FVMProblem`](@ref), it is the `(j, i)`th component that refers to the `i`th 
node of the mesh for the `j`th component of the system.
""" solve(::SteadyFVMProblem, ::Any; kwargs...)