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
    #jac_sparsity = DelaunayTriangulation.adjacency(DelaunayTriangulation.graph(DG)) + I
    # ^ The above creates a dense matrix. We can be a bit smart about this and create the 
    #   coordinate format directly for the sparse matrix conversion.
    has_ghost_edges = DelaunayTriangulation.BoundaryIndex ∈ DelaunayTriangulation.graph(DG).V
    num_nnz = 2(length(edges(DG)) - num_boundary_edges(prob) * has_ghost_edges) + num_points(prob) # Logic: For each edge in the triangulation we obtain two non-zero entries. If each boundary edge is adjoined with a ghost edge, though, then we need to make sure we don't count the contributions from those edges - hence why we subtract it off. Finally, the Jacobian needs to also include the node's relationship with itself, so we add on the number of points.
    I = zeros(Int64, num_nnz)   # row indices 
    J = zeros(Int64, num_nnz)   # col indices 
    V = ones(num_nnz)           # values (all 1)
    ctr = 1
    for i in DelaunayTriangulation._eachindex(get_points(prob))
        I[ctr] = i
        J[ctr] = i
        ctr += 1
        ngh = DelaunayTriangulation.get_neighbour(DG, i)
        for j in ngh
            if has_ghost_edges && j ≠ DelaunayTriangulation.BoundaryIndex
                I[ctr] = i
                J[ctr] = j
                ctr += 1
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
- `parallel=false`: Whether to use multithreading for evaluating the equations. Not currently used.
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
        p = (
            prob,
            collect(get_dudt_nodes(prob)),
            collect(get_interior_or_neumann_nodes(prob)),
            collect(get_boundary_elements(prob)),
            collect(get_interior_elements(prob)),
            collect(get_elements(prob)),
            collect(get_dirichlet_nodes(prob))
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
- `alg`: The algorithm to use for solving. See the DifferentialEquations.jl documentation for this.

# Keyword Arguments 
- `cache_eltype::Type{F}=eltype(get_initial_condition(prob))`: The element type used for the cache vectors. 
- `jac_prototype=float.(jacobian_sparsity(prob))`: The prototype for the sparsity pattern of the Jacobian. 
- `parallel=false`: Whether to use multithreading for evaluating the equations. Not currently used.
- `specialization::Type{S}=SciMLBase.AutoSpecialize`: The specialisation level for the `ODEProblem`.
- `chunk_size=PreallocationTools.ForwardDiff.pickchunksize(length(get_initial_condition(prob))))`: The chunk size for the dual numbers used in the cache vector.
- `kwargs...`: Extra keyword arguments for `solve` as used on `ODEProblems`.

# Output 
The output is a solution struct, as returned from DifferentialEquations.jl.
"""
function SciMLBase.solve(prob::FVMProblem, alg;
    cache_eltype::Type{F}=eltype(get_initial_condition(prob)),
    jac_prototype=float.(jacobian_sparsity(prob)),
    parallel=false,
    specialization::Type{S}=SciMLBase.AutoSpecialize,
    chunk_size=PreallocationTools.ForwardDiff.pickchunksize(length(get_initial_condition(prob))),
    kwargs...) where {S,F}
    if parallel && isinplace(prob)
        throw("The flux vector must be computed out-of-place to use the parallel equations.")
    end
    no_saveat = :saveat ∉ keys(Dict(kwargs))
    ode_problem = ODEProblem(prob; cache_eltype, jac_prototype, parallel, no_saveat, specialization, chunk_size)
    sol = solve(ode_problem, alg; kwargs...)
    return sol
end