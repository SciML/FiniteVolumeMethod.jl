function get_multithreading_parameters(prob::Union{FVMProblem,FVMSystem})
    u = prob.initial_condition
    nt = Threads.nthreads()
    if prob isa FVMProblem
        duplicated_du = DiffCache(similar(u, length(u), nt))
        dirichlet_nodes = collect(keys(get_dirichlet_nodes(prob)))
    else
        duplicated_du = DiffCache(similar(u, size(u, 1), size(u, 2), nt))
        dirichlet_nodes = ntuple(i -> collect(keys(get_dirichlet_nodes(prob, i))), _neqs(prob))
    end
    solid_triangles = collect(each_solid_triangle(prob.mesh.triangulation))
    solid_vertices = collect(each_solid_vertex(prob.mesh.triangulation))
    chunked_solid_triangles = chunks(solid_triangles, nt)
    boundary_edges = collect(keys(get_boundary_edge_map(prob.mesh.triangulation)))
    chunked_boundary_edges = chunks(boundary_edges, nt)
    return (
        duplicated_du=duplicated_du,
        dirichlet_nodes=dirichlet_nodes,
        solid_triangles=solid_triangles,
        solid_vertices=solid_vertices,
        chunked_solid_triangles=chunked_solid_triangles,
        boundary_edges=boundary_edges,
        chunked_boundary_edges=chunked_boundary_edges,
        parallel=Val(true),
        prob=prob
    )
end

function get_serial_parameters(prob::Union{FVMProblem,FVMSystem})
    return (
        parallel=Val(false),
        prob=prob
    )
end

function get_fvm_parameters(prob::Union{FVMProblem,FVMSystem}, parallel::Val{B}) where {B}
    if B
        return get_multithreading_parameters(prob)
    else
        return get_serial_parameters(prob)
    end
end

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
        for j in get_neighbours(tri, i)
            DelaunayTriangulation.is_boundary_index(j) && continue
            push!(I, i)
            push!(J, j)
            push!(V, 1.0)
        end
    end
    return sparse(I, J, V)
end

# Some working:
# Suppose we have a problem that looks like this (↓ vars → node) 
#   u₁¹     u₂¹     u₃¹     ⋯       uₙ¹ 
#   u₁²     u₂²     u₃²     ⋯       uₙ²
#    ⋮        ⋮       ⋮       ⋱       ⋮
#   u₁ᴺ     u₂ᴺ     u₃ᴺ     ⋯       uₙᴺ
# When we write down the relationships here, we need 
# to use the linear subscripts, so that the problem above 
# is interpreted as 
#   u¹      uᴺ⁺¹     u²ᴺ⁺¹     ⋯       u⁽ⁿ⁻¹⁾ᴺ⁺¹
#   u²      uᴺ⁺²     u²ᴺ⁺²     ⋯       u⁽ⁿ⁻¹⁾ᴺ⁺²
#    ⋮        ⋮       ⋮          ⋱       ⋮
#   uᴺ      u²ᴺ      u³ᴺ       ⋯       uⁿᴺ
# With this, the ith node is at the linear indices 
# (i, 1), (i, 2), …, (i, N) ↦ (i-1)*N + j for j in 1:N.
# In the original matrix, a node i being related to a node ℓ
# means that the ith and ℓ columns are all related to eachother. 
function jacobian_sparsity(prob::FVMSystem{N}) where {N}
    tri = prob.mesh.triangulation
    I = Int64[]   # row indices
    J = Int64[]   # col indices
    V = Float64[] # values (all 1)
    n = DelaunayTriangulation.num_solid_vertices(prob.mesh.triangulation_statistics)
    sizehint!(I, 6n * N) # points have, on average, six neighbours in a DelaunayTriangulation.
    sizehint!(J, 6n * N)
    sizehint!(V, 6n * N)
    for i in each_solid_vertex(tri)
        # First, i is related to itself, meaning 
        # (i, 1), (i, 2), …, (i, N) are all related. 
        for ℓ in 1:N
            node = (i - 1) * N + ℓ
            for j in 1:N
                node2 = (i - 1) * N + j
                push!(I, node)
                push!(J, node2)
                push!(V, 1.0)
            end
        end
        for j in get_neighbours(tri, i)
            DelaunayTriangulation.is_boundary_index(j) && continue
            for ℓ in 1:N
                node = (i - 1) * N + ℓ
                for k in 1:N
                    node2 = (j - 1) * N + k
                    push!(I, node)
                    push!(J, node2)
                    push!(V, 1.0)
                end
            end
        end
    end
    return sparse(I, J, V)
end
jacobian_sparsity(prob::SteadyFVMProblem) = jacobian_sparsity(prob.problem)

@inline function dirichlet_callback(has_saveat, has_dir)
    if has_dir
        cb = DiscreteCallback(
            (u, t, integrator) -> true,
            update_dirichlet_nodes!,
            save_positions=(!has_saveat, !has_saveat),
        )
    else
        cb = CallbackSet()
    end
    return cb
end

"""
    get_dirichlet_callback(prob; kwargs...)

Get the callback for updating [`Dirichlet`](@ref) nodes. The `kwargs...` argument is ignored, 
except to detect if a user has already provided a callback, in which case the 
callback gets merged into a `CallbackSet` with the [`Dirichlet`](@ref) callback. If the problem 
`prob` has no [`Dirichlet`](@ref) nodes, the returned callback does nothing and is never 
called.
"""
@inline function get_dirichlet_callback(prob; saveat=(), callback=CallbackSet(), kwargs...)
    has_dir_nodes = has_dirichlet_nodes(prob)
    dir_callback = dirichlet_callback(!isempty(saveat), has_dir_nodes)
    cb = CallbackSet(dir_callback, callback)
    return cb
end

function SciMLBase.ODEProblem(prob::Union{FVMProblem,FVMSystem};
    specialization::Type{S}=SciMLBase.AutoSpecialize,
    jac_prototype=jacobian_sparsity(prob),
    parallel::Val{B}=Val(true),
    kwargs...) where {S,B}
    initial_time = prob.initial_time
    final_time = prob.final_time
    time_span = (initial_time, final_time)
    initial_condition = prob.initial_condition
    cb = get_dirichlet_callback(prob; kwargs...)
    f = ODEFunction{true,S}(fvm_eqs!; jac_prototype)
    p = get_fvm_parameters(prob, parallel)
    ode_problem = ODEProblem{true,S}(f, initial_condition, time_span, p; callback=cb)
    return ode_problem
end

function SciMLBase.SteadyStateProblem(prob::SteadyFVMProblem;
    specialization::Type{S}=SciMLBase.AutoSpecialize,
    jac_prototype=jacobian_sparsity(prob),
    parallel::Val{B}=Val(true),
    kwargs...) where {S,B}
    ode_prob = ODEProblem(prob.problem; specialization, jac_prototype, parallel, kwargs...)
    nl_prob = SteadyStateProblem{true}(ode_prob.f, ode_prob.u0, ode_prob.p; ode_prob.kwargs...)
    return nl_prob
end

function CommonSolve.init(prob::Union{FVMProblem,FVMSystem}, args...;
    specialization::Type{S}=SciMLBase.AutoSpecialize,
    jac_prototype=jacobian_sparsity(prob),
    parallel::Val{B}=Val(true),
    kwargs...) where {S,B}
    ode_prob = SciMLBase.ODEProblem(prob; specialization, jac_prototype, parallel, kwargs...)
    return CommonSolve.init(ode_prob, args...; kwargs...)
end
function CommonSolve.solve(prob::SteadyFVMProblem, args...;
    specialization::Type{S}=SciMLBase.AutoSpecialize,
    jac_prototype=jacobian_sparsity(prob),
    parallel::Val{B}=Val(true),
    kwargs...) where {S,B}
    nl_prob = SciMLBase.SteadyStateProblem(prob; specialization, jac_prototype, parallel, kwargs...)
    return CommonSolve.solve(nl_prob, args...; kwargs...)
end


@doc """
    solve(prob::Union{FVMProblem,FVMSystem}, alg; 
        specialization=SciMLBase.AutoSpecialize, 
        jac_prototype=jacobian_sparsity(prob), 
        parallel::Val{<:Bool}=Val(true),
        kwargs...)


Solves the given [`FVMProblem`](@ref) or [`FVMSystem`](@ref) `prob` with the algorithm `alg`.

# Arguments 
- `prob`: The problem to be solved.
- `alg`: The algorithm to be used to solve the problem. This can be any of the algorithms in DifferentialEquations.jl.

# Keyword Arguments
- `specialization=SciMLBase.AutoSpecialize`: The type of specialization to be used. See https://docs.sciml.ai/DiffEqDocs/stable/features/low_dep/#Controlling-Function-Specialization-and-Precompilation.
- `jac_prototype=jacobian_sparsity(prob)`: The prototype for the Jacobian matrix, constructed by default from `jacobian_sparsity`.
- `parallel::Val{<:Bool}=Val(true)`: Whether to use multithreading. Use `Val(false)` to disable multithreading. 
- `kwargs...`: Any other keyword arguments to be passed to the solver.

# Outputs 
The returned value `sol` depends on the type of the problem.
- [`FVMProblem`](@ref)

In this case, `sol::ODESolution` is such that the `i`th component of `sol` refers to the `i`th node of the underlying mesh.
- [`FVMSystem`](@ref)

In this case, the `(j, i)`th component of `sol::ODESolution` refers to the `i`th node of the underlying mesh for the `j`th component of the system.
""" solve(::Union{FVMProblem,FVMSystem}, ::Any; kwargs...)

@doc """
    solve(prob::SteadyFVMProblem, alg; 
        specialization=SciMLBase.AutoSpecialize, 
        jac_prototype=jacobian_sparsity(prob),
        parallel::Val{<:Bool}=Val(true),
        kwargs...)


Solves the given [`FVMProblem`](@ref) or [`FVMSystem`](@ref) `prob` with the algorithm `alg`.

# Arguments 
- `prob`: The problem to be solved.
- `alg`: The algorithm to be used to solve the problem. This can be any of the algorithms in NonlinearSolve.jl.

# Keyword Arguments
- `specialization=SciMLBase.AutoSpecialize`: The type of specialization to be used. See https://docs.sciml.ai/DiffEqDocs/stable/features/low_dep/#Controlling-Function-Specialization-and-Precompilation.
- `jac_prototype=jacobian_sparsity(prob)`: The prototype for the Jacobian matrix, constructed by default from `jacobian_sparsity`.
- `parallel::Val{<:Bool}=Val(true)`: Whether to use multithreading. Use `Val(false)` to disable multithreading.
- `kwargs...`: Any other keyword arguments to be passed to the solver.

# Outputs 
The returned value `sol` depends on whether the underlying problem is a [`FVMProblem`](@ref) or an [`FVMSystem`](@ref), but in 
each case it is an `ODESolution` type that can be accessed like the solutions in DifferentialEquations.jl:
- [`FVMProblem`](@ref)

In this case, `sol` is such that the `i`th component of `sol` refers to the `i`th node of the underlying mesh.
- [`FVMSystem`](@ref)

In this case, the `(j, i)`th component of `sol` refers to the `i`th node of the underlying mesh for the `j`th component of the system.
""" solve(::SteadyFVMProblem, ::Any; kwargs...)