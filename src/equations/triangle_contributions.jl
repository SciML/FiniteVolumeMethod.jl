# get the fluxes from each centroid-edge edge
@inline function get_fluxes(prob, props, α, β, γ, t)
    q1 = get_flux(prob, props, α, β, γ, t, 1)
    q2 = get_flux(prob, props, α, β, γ, t, 2)
    q3 = get_flux(prob, props, α, β, γ, t, 3)
    return q1, q2, q3
end

# update du with the fluxes from each centroid-edge edge for a non-system
@inline function update_du!(du, prob::AbstractFVMProblem, i, j, k, summand₁, summand₂, summand₃)
    has_condition(prob, i) || (du[i] = du[i] + summand₃ - summand₁)
    has_condition(prob, j) || (du[j] = du[j] + summand₁ - summand₂)
    has_condition(prob, k) || (du[k] = du[k] + summand₂ - summand₃)
    return nothing
end

# update du with the fluxes from each centroid-edge edge for a system for all variables
@inline function update_du!(du, prob::FVMSystem, i, j, k, summand₁, summand₂, summand₃) 
    for var in 1:_neqs(prob)
        has_condition(prob, i, var) || (du[var, i] = du[var, i] + summand₃[var] - summand₁[var])
        has_condition(prob, j, var) || (du[var, j] = du[var, j] + summand₁[var] - summand₂[var])
        has_condition(prob, k, var) || (du[var, k] = du[var, k] + summand₂[var] - summand₃[var])
    end
    return nothing
end

# get the contributions to the dudt system across a single triangle
@inline function fvm_eqs_single_triangle!(du, u, prob, t, T)
    i, j, k = indices(T)
    props = get_triangle_props(prob, i, j, k)
    α, β, γ = get_shape_function_coefficients(props, T, u, prob)
    summand₁, summand₂, summand₃ = get_fluxes(prob, props, α, β, γ, t)
    update_du!(du, prob, i, j, k, summand₁, summand₂, summand₃)
    return nothing
end

# get the contributions to the dudt system across all triangles
function get_triangle_contributions!(du, u, prob, t)
    for T in each_solid_triangle(prob.mesh.triangulation)
        fvm_eqs_single_triangle!(du, u, prob, t, T)
    end
    return nothing
end

# get the contributions to the dudt system across all triangles in parallel
function get_parallel_triangle_contributions!(duplicated_du, u, prob, t, chunked_solid_triangles, solid_triangles)
    Threads.@threads for (triangle_range, chunk_idx) in chunked_solid_triangles
        _get_parallel_triangle_contributions!(duplicated_du, u, prob, t, triangle_range, chunk_idx, solid_triangles)
    end
    return nothing
end

# get the contributions to the dudt system across a chunk of triangles
function _get_parallel_triangle_contributions!(duplicated_du, u, prob, t, triangle_range, chunk_idx, solid_triangles)
    for triangle_idx in triangle_range
        T = solid_triangles[triangle_idx]
        if prob isa FVMSystem
            @views fvm_eqs_single_triangle!(duplicated_du[:, :, chunk_idx], u, prob, t, T)
        else
            @views fvm_eqs_single_triangle!(duplicated_du[:, chunk_idx], u, prob, t, T)
        end
    end
    return nothing
end