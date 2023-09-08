@inline function get_fluxes(prob, props, α, β, γ, t)
    q1 = get_flux(prob, props, α, β, γ, t, 1)
    q2 = get_flux(prob, props, α, β, γ, t, 2)
    q3 = get_flux(prob, props, α, β, γ, t, 3)
    return q1, q2, q3
end

@inline function update_du!(du, ::AbstractFVMProblem, i, j, k, summand₁, summand₂, summand₃)
    du[i] = du[i] + summand₃ - summand₁
    du[j] = du[j] + summand₁ - summand₂
    du[k] = du[k] + summand₂ - summand₃
    return nothing
end
@inline function update_du!(du, ::FVMSystem{N}, i, j, k, ℓ, summand₁, summand₂, summand₃) where {N}
    for var in 1:N
        du[var, i] = du[var, i] + summand₃[var] - summand₁[var]
        du[var, j] = du[var, j] + summand₁[var] - summand₂[var]
        du[var, k] = du[var, k] + summand₂[var] - summand₃[var]
    end
    return nothing
end

@inline function fvm_eqs_single_triangle!(du, u, prob, t, T)
    i, j, k = indices(T)
    props = get_triangle_props(prob, i, j, k)
    α, β, γ = get_shape_function_coefficients(props, T, u, prob)
    summand₁, summand₂, summand₃ = get_fluxes(prob, props, α, β, γ, t)
    update_du!(du, prob, i, j, k, summand₁, summand₂, summand₃)
    return nothing
end

function get_triangle_contributions!(du, u, prob, t)
    for T in each_solid_triangle(prob.mesh.triangulation)
        fvm_eqs_single_triangle!(du, u, prob, t, T)
    end
    return nothing
end

function get_parallel_triangle_contributions!(duplicated_du, u, prob, t, chunked_solid_triangles, solid_triangles)
    Threads.@threads for (triangle_range, chunk_idx) in chunked_solid_triangles
        _get_parallel_triangle_contributions!(duplicated_du, u, prob, t, triangle_range, chunk_idx, solid_triangles)
    end
    return nothing
end
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