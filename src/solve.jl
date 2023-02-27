@inline function dirichlet_callback(no_saveat=false)
    condition = (u, t, integrator) -> true
    cb = DiffEqBase.DiscreteCallback(condition, update_dirichlet_nodes!; save_positions=(no_saveat, no_saveat))
    return cb
end

"""
    jacobian_sparsity(prob::FVMProblem)

Constructs the sparse matrix which has the same sparsity pattern as the Jacobian for the finite volume equations 
corresponding to the [`FVMProblem`](@ref) given by `prob`.
"""
function jacobian_sparsity(prob::FVMProblem)
    DG = get_neighbours(prob)
    I = Int64[]   # row indices 
    J = Int64[]   # col indices 
    V = Float64[] # values (all 1)
    for i in each_point_index(prob)
        push!(I, i)
        push!(J, i)
        push!(V, 1.0)
        ngh = get_neighbours(DG, i)
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

"""
    SciMLBase.ODEProblem(prob::FVMProblem;
        cache_eltype::Type{F}=eltype(get_initial_condition(prob)),
        jac_prototype=float.(jacobian_sparsity(prob)),
        parallel=false,
        no_saveat=true,
        specialization::Type{S}=SciMLBase.AutoSpecialize,
        chunk_size=PreallocationTools.ForwardDiff.pickchunksize(length(get_initial_condition(prob)))) where {S,F}

Constructs the `ODEProblem` for the system of ODEs defined by the finite volume equations corresponding to `prob`.

# Arguments 
- `prob::FVMProblem`: The [`FVMProblem`](@ref).

# Keyword Arguments 
- `cache_eltype::Type{F}=eltype(get_initial_condition(prob))`: The element type used for the cache vectors. 
- `jac_prototype=float.(jacobian_sparsity(prob))`: The prototype for the sparsity pattern of the Jacobian. 
- `parallel=false`: Whether to use multithreading for evaluating the equations.
- `no_saveat=true`: Whether the solution is saving at specific points. 
- `specialization::Type{S}=SciMLBase.AutoSpecialize`: The specialisation level for the `ODEProblem`.
- `chunk_size=PreallocationTools.ForwardDiff.pickchunksize(length(get_initial_condition(prob))))`: The chunk size for the dual numbers used in the cache vector.

# Outputs 
Returns the corresponding [`ODEProblem`](@ref).
"""
function SciMLBase.ODEProblem(prob::FVMProblem;
    cache_eltype::Type{F}=eltype(get_initial_condition(prob)),
    jac_prototype=float.(jacobian_sparsity(prob)),
    parallel=false,
    no_saveat=true,
    specialization::Type{S}=SciMLBase.AutoSpecialize,
    chunk_size=PreallocationTools.ForwardDiff.pickchunksize(length(get_initial_condition(prob))),
    kwargs...) where {S,F}
    time_span = get_time_span(prob)
    initial_condition = get_initial_condition(prob)
    cb = dirichlet_callback(no_saveat)
    if !parallel
        flux_cache = PreallocationTools.DiffCache(zeros(F, 2), chunk_size)
        shape_coeffs = PreallocationTools.DiffCache(zeros(F, 3), chunk_size)
        f = ODEFunction{true,S}(fvm_eqs!; jac_prototype)
        p = (prob, flux_cache, shape_coeffs)
        ode_problem = ODEProblem{true,S}(f, initial_condition, time_span, p; callback=cb, kwargs...)
        return ode_problem
    else
        f = ODEFunction{true,S}(par_fvm_eqs!; jac_prototype)
        u0 = get_initial_condition(prob)
        du_copies,
        flux_caches,
        shape_coeffs,
        dudt_nodes,
        interior_or_neumann_nodes,
        boundary_elements,
        interior_elements,
        elements,
        dirichlet_nodes,
        chunked_boundary_elements,
        chunked_interior_elements,
        chunked_elements = prepare_vectors_for_multithreading(u0, prob, F; chunk_size)
        p = (
            prob,
            du_copies,
            flux_caches,
            shape_coeffs,
            dudt_nodes,
            interior_or_neumann_nodes,
            boundary_elements,
            interior_elements,
            elements,
            dirichlet_nodes,
            chunked_boundary_elements,
            chunked_interior_elements,
            chunked_elements
        )
        ode_problem = ODEProblem{true,S}(f, initial_condition, time_span, p; callback=cb, kwargs...)
        return ode_problem
    end
end

"""
    SciMLBase.solve(prob::FVMProblem, alg;
        cache_eltype::Type{F}=eltype(get_initial_condition(prob)),
        jac_prototype=float.(jacobian_sparsity(prob)),
        parallel=false,
        specialization::Type{S}=SciMLBase.AutoSpecialize,
        chunk_size=PreallocationTools.ForwardDiff.pickchunksize(length(get_initial_condition(prob))),
        kwargs...)

Solves the [`FVMProblem`](@ref) given by `prob` using the algorithm `alg`.

# Arguments 
- `prob::FVMProblem`: The [`FVMProblem`](@ref).
- `alg`: The algorithm to use for solving. See the DifferentialEquations.jl documentation for this. If your problem is a steady problem, then you should use an algorithm from NonlinearSolve.jl instead - note that the initial estimate in this case comes from the initial condition.

# Keyword Arguments 
- `cache_eltype::Type{F}=eltype(get_initial_condition(prob))`: The element type used for the cache vectors. 
- `jac_prototype=float.(jacobian_sparsity(prob))`: The prototype for the sparsity pattern of the Jacobian. 
- `parallel=false`: Whether to use multithreading for evaluating the equations. Not currently used.
- `specialization::Type{S}=SciMLBase.AutoSpecialize`: The specialisation level for the `ODEProblem`.
- `chunk_size=PreallocationTools.ForwardDiff.pickchunksize(length(get_initial_condition(prob))))`: The chunk size for the dual numbers used in the cache vector.
- `kwargs...`: Extra keyword arguments for `solve` as used on `ODEProblem`s (or `NonlinearProblem`s if your problem is steady).

# Output 
The output is a solution struct, as returned from DifferentialEquations.jl. If instead your problem is steady, returns a 
solution struct as returned from NonlinearSolve.jl.
"""
function SciMLBase.solve(prob::FVMProblem, alg;
    cache_eltype::Type{F}=eltype(get_initial_condition(prob)),
    jac_prototype=float.(jacobian_sparsity(prob)),
    parallel=false,
    specialization::Type{S}=SciMLBase.AutoSpecialize,
    chunk_size=PreallocationTools.ForwardDiff.pickchunksize(length(get_initial_condition(prob))),
    kwargs...) where {S,F}
    no_saveat = :saveat ∉ keys(Dict(kwargs))
    ode_problem = ODEProblem(prob; cache_eltype, jac_prototype, parallel, no_saveat, specialization, chunk_size)
    if !is_steady(prob)
        sol = solve(ode_problem, alg; kwargs...)
        return sol
    else
        nonlinear_problem = NonlinearProblem{true}(ode_problem.f, ode_problem.u0, ode_problem.p)
        sol = solve(nonlinear_problem, alg; kwargs...)
        return sol
    end
end