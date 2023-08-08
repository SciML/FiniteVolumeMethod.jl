@inline getα(shape_coeffs) =  shape_coeffs[1]
@inline getβ(shape_coeffs) =  shape_coeffs[2]
@inline getγ(shape_coeffs) =  shape_coeffs[3]

@inline function linear_shape_function_coefficients!(shape_coeffs, u, prob, T)
    i, j, k = indices(T)
    @muladd  shape_coeffs[1] = gets(prob, T, 1) * u[i] + gets(prob, T, 2) * u[j] + gets(prob, T, 3) * u[k]
    @muladd  shape_coeffs[2] = gets(prob, T, 4) * u[i] + gets(prob, T, 5) * u[j] + gets(prob, T, 6) * u[k]
    @muladd  shape_coeffs[3] = gets(prob, T, 7) * u[i] + gets(prob, T, 8) * u[j] + gets(prob, T, 9) * u[k]
    return nothing
end

function fvm_eqs_edge!(du, t, (vj, j), (vjnb, jnb), α, β, γ, prob, flux_cache, T)
    x, y = get_control_volume_edge_midpoints(prob, T, j)
    xn, yn = get_normals(prob, T, j)
    ℓ = get_lengths(prob, T, j)
    if isinplace(prob)
        get_flux!(flux_cache, prob, x, y, t, α, β, γ)
    else
        flux_cache = get_flux(prob, x, y, t, α, β, γ) # Make no assumption that flux_cache is mutable 
    end
    @muladd  summand = -(getx(flux_cache) * xn + gety(flux_cache) * yn) * ℓ
    if is_interior_or_neumann_node(prob, vj)
         du[vj] += summand
    end
    if is_interior_or_neumann_node(prob, vjnb)
         du[vjnb] -= summand
    end
    return nothing
end
@inline function fvm_eqs_edge!(du, t, α, β, γ, prob, flux_cache, T)
    i, j, k = indices(T)
    fvm_eqs_edge!(du, t, (i, 1), (j, 2), α, β, γ, prob, flux_cache, T)#unrolled
    fvm_eqs_edge!(du, t, (j, 2), (k, 3), α, β, γ, prob, flux_cache, T)
    fvm_eqs_edge!(du, t, (k, 3), (i, 1), α, β, γ, prob, flux_cache, T)
    return nothing
end

@inline function fvm_eqs_interior_element!(du, u, t, prob, shape_coeffs, flux_cache, T)
    linear_shape_function_coefficients!(shape_coeffs, u, prob, T)
    α = getα(shape_coeffs)
    β = getβ(shape_coeffs)
    γ = getγ(shape_coeffs)
    fvm_eqs_edge!(du, t, α, β, γ, prob, flux_cache, T)
    return nothing
end
@inline function fvm_eqs_interior_element!(du, u, t, prob, shape_coeffs, flux_cache)
    for V in get_interior_elements(prob)
        fvm_eqs_interior_element!(du, u, t, prob, shape_coeffs, flux_cache, V)
    end
    return nothing
end

@inline function fvm_eqs_boundary_element!(du, u, t, prob, shape_coeffs, flux_cache, T)
    linear_shape_function_coefficients!(shape_coeffs, u, prob, T)
    α = getα(shape_coeffs)
    β = getβ(shape_coeffs)
    γ = getγ(shape_coeffs)
    interior_edges = get_interior_edges(prob, T)
    for ((vj, j), (vjnb, jnb)) in interior_edges
        fvm_eqs_edge!(du, t, (vj, j), (vjnb, jnb), α, β, γ, prob, flux_cache, T)
    end
    return nothing
end
@inline function fvm_eqs_boundary_element!(du, u, t, prob, shape_coeffs, flux_cache)
    for V in get_boundary_elements(prob)
        fvm_eqs_boundary_element!(du, u, t, prob, shape_coeffs, flux_cache, V)
    end
    return nothing
end

@inline function fvm_eqs_source_contribution!(du, u, t, j, prob)
    x, y = get_point(prob, j)
    V = get_volumes(prob, j)
     R = get_reaction(prob, x, y, t, u[j])
    @muladd  du[j] = du[j] / V + R
    return nothing
end
@inline function fvm_eqs_source_contribution!(du, u, t, prob)
    for j ∈ get_interior_or_neumann_nodes(prob)
        fvm_eqs_source_contribution!(du, u, t, j, prob)
    end
    return nothing
end

@inline function evaluate_boundary_function(u, t, j, prob)
    segment_number = map_node_to_segment(prob, j)
    x, y = get_point(prob, j)
     val = evaluate_boundary_function(prob, segment_number, x, y, t, u[j])
    return val
end
@inline function evaluate_boundary_function!(du, u, t, j, prob)
     du[j] = evaluate_boundary_function(u, t, j, prob)
    return nothing
end

@inline function update_dudt_nodes!(du, u, t, prob)
    for j in get_dudt_nodes(prob)
        evaluate_boundary_function!(du, u, t, j, prob)
    end
    return nothing
end

function fvm_eqs!(du::AbstractVector{T}, u, p, t) where {T}
    prob, flux_cache, shape_coeffs = p
    tmp_flux_cache = get_tmp(flux_cache, u)
    tmp_shape_coeffs = get_tmp(shape_coeffs, u)
    fill!(du, zero(T))
    fvm_eqs_interior_element!(du, u, t, prob, tmp_shape_coeffs, tmp_flux_cache)
    fvm_eqs_boundary_element!(du, u, t, prob, tmp_shape_coeffs, tmp_flux_cache)
    fvm_eqs_source_contribution!(du, u, t, prob)
    update_dudt_nodes!(du, u, t, prob)
    return nothing
end

function update_dirichlet_nodes!(u, t, prob)
    for j in get_dirichlet_nodes(prob)
         u[j] = evaluate_boundary_function(u, t, j, prob)
    end
    return nothing
end
function update_dirichlet_nodes!(integrator)
     update_dirichlet_nodes!(integrator.u, integrator.t, integrator.p[1])
    return nothing
end