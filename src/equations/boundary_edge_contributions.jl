# primitive: get flux contribution across a boundary edge (i, j), taking care for a Neumann boundary condition
@inline function _get_boundary_flux(prob::AbstractFVMProblem, x, y, t, α, β, γ, nx, ny, i, j, u::T) where {T}
    # For checking if an edge is Neumann, we need only check e.g. (i, j) and not (j, i), since we do not allow for internal Neumann edges.
    ij_is_neumann = is_neumann_edge(prob, i, j)
    qn = if !ij_is_neumann
        _non_neumann_get_flux(prob, x, y, t, α, β, γ, nx, ny)
    else
        _neumann_get_flux(prob, x, y, t, u, i, j)
    end
    return qn
end

# primitive: get flux contribution across a boundary edge (i, j) in a system, taking care for a Neumann boundary condition for a single variable. This is used as a function barrier
@inline function _get_boundary_flux(prob::FVMSystem, x, y, t, nx, ny, i, j, ℓ, u::T, var, flux) where {T}
    ij_is_neumann = is_neumann_edge(prob, i, j, var)
    if !ij_is_neumann
        _qx, _qy = flux[var]
        qn = (_qx * nx + _qy * ny) * ℓ
    else
        qn = _neumann_get_flux(prob, x, y, t, u, i, j, var) * ℓ
    end
    return qn
end

# get flux contribution across a boundary edge (i, j), taking care for a Neumann boundary condition for all variables in a system
@inline function _get_boundary_fluxes(prob::FVMSystem, x, y, t, α, β, γ, nx, ny, i, j, u::T, ℓ) where {T}
    all_flux = eval_flux_function(prob, x, y, t, α, β, γ) # if we use the primitive, then we need to recursively evaluate the flux over and over
    normal_flux = ntuple(_neqs(prob)) do var
        _get_boundary_flux(prob, x, y, t, nx, ny, i, j, ℓ, u, var, all_flux)
    end
    return normal_flux
end

# function for getting both fluxes for a non-system problem
@inline function get_boundary_fluxes(prob::AbstractFVMProblem, α::T, β, γ, i, j, t) where {T}
    nx, ny, mᵢx, mᵢy, mⱼx, mⱼy, ℓ = get_boundary_cv_components(prob.mesh, i, j)
    q1 = _get_boundary_flux(prob, mᵢx, mᵢy, t, α, β, γ, nx, ny, i, j, α * mᵢx + β * mᵢy + γ)
    q2 = _get_boundary_flux(prob, mⱼx, mⱼy, t, α, β, γ, nx, ny, i, j, α * mⱼx + β * mⱼy + γ)
    return q1 * ℓ, q2 * ℓ
end

# function for getting both fluxes for a system problem
@inline function get_boundary_fluxes(prob::FVMSystem, α::T, β, γ, i, j, t) where {T}
    nx, ny, mᵢx, mᵢy, mⱼx, mⱼy, ℓ = get_boundary_cv_components(prob.mesh, i, j)
    u_shapeᵢ = ntuple(var -> α[var] * mᵢx + β[var] * mᵢy + γ[var], _neqs(prob))
    u_shapeⱼ = ntuple(var -> α[var] * mⱼx + β[var] * mⱼy + γ[var], _neqs(prob))
    q1 = _get_boundary_fluxes(prob, mᵢx, mᵢy, t, α, β, γ, nx, ny, i, j, u_shapeᵢ, ℓ)
    q2 = _get_boundary_fluxes(prob, mⱼx, mⱼy, t, α, β, γ, nx, ny, i, j, u_shapeⱼ, ℓ)
    return q1, q2
end

# function for applying both fluxes for a non-system problem
@inline function update_du!(du, ::AbstractFVMProblem, i, j, summand₁, summand₂)
    du[i] = du[i] - summand₁
    du[j] = du[j] - summand₂
    return nothing
end

# function for applying both fluxes for a system problem
@inline function update_du!(du, prob::FVMSystem, i, j, summand₁, summand₂)
    for var in 1:_neqs(prob)
        du[var, i] = du[var, i] - summand₁[var]
        du[var, j] = du[var, j] - summand₂[var]
    end
    return nothing
end

# get the contributions to the dudt system across a boundary edge
@inline function fvm_eqs_single_boundary_edge!(du, u, prob, t, e)
    i, j = DelaunayTriangulation.edge_vertices(e)
    k = get_adjacent(prob.mesh.triangulation, e)
    T = (i, j, k)
    T, props = _safe_get_triangle_props(prob, T)
    α, β, γ = get_shape_function_coefficients(props, T, u, prob)
    summand₁, summand₂ = get_boundary_fluxes(prob, α, β, γ, i, j, t)
    update_du!(du, prob, i, j, summand₁, summand₂)
    return nothing
end

# get the contributions to the dudt system across all boundary edges
function get_boundary_edge_contributions!(du, u, prob, t)
    for e in keys(get_boundary_edge_map(prob.mesh.triangulation))
        fvm_eqs_single_boundary_edge!(du, u, prob, t, e)
    end
    return nothing
end

# get the contributions to the dudt system across all boundary edges in parallel
function get_parallel_boundary_edge_contributions!(duplicated_du, u, prob, t, chunked_boundary_edges, boundary_edges)
    Threads.@threads for (edge_range, chunk_idx) in chunked_boundary_edges
        _get_parallel_boundary_edge_contributions!(duplicated_du, u, prob, t, edge_range, chunk_idx, boundary_edges)
    end
    return nothing
end

# get the contributions to the dudt system across a chunk of boundary edges
function _get_parallel_boundary_edge_contributions!(duplicated_du, u, prob, t, edge_range, chunk_idx, boundary_edges)
    for edge_idx in edge_range
        e = boundary_edges[edge_idx]
        if prob isa FVMSystem
            @views fvm_eqs_single_boundary_edge!(duplicated_du[:, :, chunk_idx], u, prob, t, e)
        else
            @views fvm_eqs_single_boundary_edge!(duplicated_du[:, chunk_idx], u, prob, t, e)
        end
    end
end
