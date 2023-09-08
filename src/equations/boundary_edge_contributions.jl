# function for getting both fluxes for a non-system problem
@inline function get_boundary_fluxes(prob::AbstractFVMProblem, α::T, β, γ, i, j, t) where {T}
    nx, ny, mᵢx, mᵢy, mⱼx, mⱼy, ℓᵢ, ℓⱼ = _get_boundary_cv_components(prob, i, j)
    q1 = _get_boundary_flux(prob, mᵢx, mᵢy, t, α, β, γ, nx, ny, i, j, α * mᵢx + β * mᵢy + γ) * one(eltype(T))
    q2 = _get_boundary_flux(prob, mⱼx, mⱼy, t, α, β, γ, nx, ny, i, j, α * mⱼx + β * mⱼy + γ) * one(eltype(T))
    return q1 * ℓᵢ, q2 * ℓⱼ
end

# function for getting both fluxes for a system problem
@inline function get_boundary_fluxes(prob::FVMSystem{N}, α::T, β, γ, i, j, t) where {N,T}
    nx, ny, mᵢx, mᵢy, mⱼx, mⱼy, ℓᵢ, ℓⱼ = _get_boundary_cv_components(prob, i, j)
    u_shapeᵢ = ntuple(var -> α[var] * mᵢx + β[var] * mᵢy + γ[var], Val(N))
    u_shapeⱼ = ntuple(var -> α[var] * mⱼx + β[var] * mⱼy + γ[var], Val(N))
    q1 = _get_boundary_fluxes(prob, mᵢx, mᵢy, t, α, β, γ, nx, ny, i, j, u_shapeᵢ, ℓᵢ)
    q2 = _get_boundary_fluxes(prob, mⱼx, mⱼy, t, α, β, γ, nx, ny, i, j, u_shapeⱼ, ℓⱼ)
    return q1, q2
end

# function for applying both fluxes for a non-system problem
@inline function update_du!(du, ::AbstractFVMProblem, i, j, summand₁, summand₂)
    du[i] = du[i] - summand₁
    du[j] = du[j] - summand₂
    return nothing
end

# function for applying both fluxes for a system problem
@inline function update_du!(du, ::FVMSystem{N}, i, j, ℓ, summand₁, summand₂) where {N}
    for var in 1:N
        du[var, i] = du[var, i] - summand₁[var]
        du[var, j] = du[var, j] - summand₂[var]
    end
    return nothing
end

# get the contributions to the dudt system across a boundary edge
@inline function fvm_eqs_single_boundary_edge!(du, u, prob, t, e)
    i, j = DelaunayTriangulation.edge_indices(e)
    k = get_adjacent(prob.mesh.triangulation, e)
    props = get_triangle_props(prob, i, j, k)
    T = (i, j, k)
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
