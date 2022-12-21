@inbounds @muladd getα(prob, T, u) = gets(prob, T, 1) * u[T[1]] + gets(prob, T, 2) * u[T[2]] + gets(prob, T, 3) * u[T[3]]
@inbounds @muladd getβ(prob, T, u) = gets(prob, T, 4) * u[T[1]] + gets(prob, T, 5) * u[T[2]] + gets(prob, T, 6) * u[T[3]]
@inbounds @muladd getγ(prob, T, u) = gets(prob, T, 7) * u[T[1]] + gets(prob, T, 8) * u[T[2]] + gets(prob, T, 9) * u[T[3]]

function par_fvm_eqs_edge!(du, t, (vj, j), (vjnb, jnb), α, β, γ, prob, flux_cache, T)
    x, y = get_control_volume_edge_midpoints(prob, T, j)
    xn, yn = get_normals(prob, T, j)
    ℓ = get_lengths(prob, T, j)
    if isinplace(prob)
        get_flux!(flux_cache, prob, x, y, t, α, β, γ)
    else
        flux_cache = get_flux(prob, x, y, t, α, β, γ) # Make no assumption that flux_cache is mutable 
    end
    @muladd @inbounds summand = -(getx(flux_cache) * xn + gety(flux_cache) * yn) * ℓ
    if is_interior_or_neumann_node(prob, vj)
        @inbounds du[vj] += summand
    end
    if is_interior_or_neumann_node(prob, vjnb)
        @inbounds du[vjnb] -= summand
    end
    return nothing
end
@inline function par_fvm_eqs_edge!(du, t, α, β, γ, prob, flux_cache, T)
    i, j, k = indices(T)
    par_fvm_eqs_edge!(du, t, (i, 1), (j, 2), α, β, γ, prob, flux_cache, T)#unrolled
    par_fvm_eqs_edge!(du, t, (j, 2), (k, 3), α, β, γ, prob, flux_cache, T)
    par_fvm_eqs_edge!(du, t, (k, 3), (i, 1), α, β, γ, prob, flux_cache, T)
    return nothing
end

@inline function par_fvm_eqs_interior_element!(du, u, t, prob, shape_coeffs, flux_cache, T)
    linear_shape_function_coefficients!(shape_coeffs, u, prob, T)
    α = getα(shape_coeffs)
    β = getβ(shape_coeffs)
    γ = getγ(shape_coeffs)
    par_fvm_eqs_edge!(du, t, α, β, γ, prob, flux_cache, T)
    return nothing
end
@inline function par_fvm_eqs_interior_element!(du, u::AbstractVector{T}, t, prob, interior_elements, chunked_interior_elements, flux_caches, shape_coeffs) where {T}
    Threads.@threads for (linear_triangle_idx, chunk_idx) in chunked_interior_elements
        for V in @views interior_elements[linear_triangle_idx]
            tmp_flux_cache = get_tmp(flux_caches[chunk_idx], u)
            tmp_shape_coeffs = get_tmp(shape_coeffs[chunk_idx], u)
            @views par_fvm_eqs_interior_element!(du[:, chunk_idx], u, t, prob, tmp_shape_coeffs, tmp_flux_cache, V)
        end
    end
    return nothing
end

@inline function par_fvm_eqs_boundary_element!(du, u, t, prob, T, flux_cache, shape_coeffs)
    linear_shape_function_coefficients!(shape_coeffs, u, prob, T)
    α = getα(shape_coeffs)
    β = getβ(shape_coeffs)
    γ = getγ(shape_coeffs)
    interior_edges = get_interior_edges(prob, T)
    for ((vj, j), (vjnb, jnb)) in interior_edges
        par_fvm_eqs_edge!(du, t, (vj, j), (vjnb, jnb), α, β, γ, prob, flux_cache, T)
    end
    return nothing
end
@inline function par_fvm_eqs_boundary_element!(du, u::AbstractVector{T}, t, prob, boundary_elements, chunked_boundary_elements, flux_caches, shape_coeffs) where {T}
    Threads.@threads for (linear_triangle_idx, chunk_idx) in chunked_boundary_elements
        for V in @views boundary_elements[linear_triangle_idx]
            tmp_flux_cache = get_tmp(flux_caches[chunk_idx], u)
            tmp_shape_coeffs = get_tmp(shape_coeffs[chunk_idx], u)
            @views par_fvm_eqs_boundary_element!(du[:, chunk_idx], u, t, prob, V, tmp_flux_cache, tmp_shape_coeffs)
        end
    end
    return nothing
end

@inline function par_fvm_eqs_source_contribution!(du, u, t, j::Integer, prob)
    x, y = get_point(prob, j)
    V = get_volumes(prob, j)
    @inbounds R = get_reaction(prob, x, y, t, u[j])
    @inbounds @muladd du[j] = du[j] / V + R
    return nothing
end
@inline function par_fvm_eqs_source_contribution!(du, u, t, prob, interior_or_neumann_nodes)
    Threads.@threads for j in interior_or_neumann_nodes
        par_fvm_eqs_source_contribution!(du, u, t, j, prob)
    end
    return nothing
end

@inline function par_update_dudt_node!(du, u, t, prob, dudt_nodes)
    Threads.@threads for j in dudt_nodes 
        evaluate_boundary_function!(du, u, t, j, prob)
    end
    return nothing
end

function par_fvm_eqs!(du::AbstractVector{T}, u, p, t) where {T}
    prob,#1
    du_copies,#2
    flux_caches,#3
    shape_coeffs,#4
    dudt_nodes,#5
    interior_or_neumann_nodes,#6
    boundary_elements,#7
    interior_elements,#8
    elements,#9
    dirichlet_nodes,#10
    chunked_boundary_elements,#11
    chunked_interior_elements,
    chunked_elements = p
    fill!(du, zero(T))
    duplicated_du = get_tmp(du_copies, du)
    fill!(duplicated_du, zero(T))
    par_fvm_eqs_interior_element!(duplicated_du, u, t, prob, interior_elements, chunked_interior_elements, flux_caches, shape_coeffs)
    par_fvm_eqs_boundary_element!(duplicated_du, u, t, prob, boundary_elements, chunked_boundary_elements, flux_caches, shape_coeffs)
    for _du in eachcol(duplicated_du)
        du .+= _du
    end
    par_fvm_eqs_source_contribution!(du, u, t, prob, interior_or_neumann_nodes)
    par_update_dudt_node!(du, u, t, prob, dudt_nodes)
    return nothing
end

function par_update_dirichlet_nodes!(u, t, prob, dirichlet_nodes)
    Threads.@threads for j in dirichlet_nodes
        @inbounds u[j] = evaluate_boundary_function(u, t, j, prob)
    end
    return nothing
end
function par_update_dirichlet_nodes!(integrator)
    @inbounds par_update_dirichlet_nodes!(integrator.u, integrator.t, integrator.p[1], integrator.p[10])
    return nothing
end

function prepare_vectors_for_multithreading(u, prob, cache_eltype::Type{F};
    chunk_size=PreallocationTools.ForwardDiff.pickchunksize(length(u))) where {F}
    nt = Threads.nthreads()
    du = DiffCache(zeros(F, length(u), nt), chunk_size)
    flux_caches = [DiffCache(zeros(F, 2), chunk_size) for _ in 1:nt]
    shape_coeffs = [DiffCache(zeros(F, 3), chunk_size) for _ in 1:nt]
    dudt_nodes = get_dudt_nodes(prob) |> collect
    interior_or_neumann_nodes = get_interior_or_neumann_nodes(prob) |> collect
    boundary_elements = get_boundary_elements(prob) |> collect
    interior_elements = get_interior_elements(prob) |> collect
    elements = get_elements(prob) |> collect # collects are needed so that firstindex is defined for Threads
    dirichlet_nodes = get_dirichlet_nodes(prob) |> collect
    chunked_boundary_elements = chunks(boundary_elements, nt)
    chunked_interior_elements = chunks(interior_elements, nt)
    chunked_elements = chunks(elements, nt)
    return du,
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
end