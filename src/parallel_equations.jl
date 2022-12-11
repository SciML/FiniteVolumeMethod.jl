const fvm_splock = Base.Threads.SpinLock()

@inbounds @muladd getα(prob, T, u) = gets(prob, T, 1) * u[T[1]] + gets(prob, T, 2) * u[T[2]] + gets(prob, T, 3) * u[T[3]]
@inbounds @muladd getβ(prob, T, u) = gets(prob, T, 4) * u[T[1]] + gets(prob, T, 5) * u[T[2]] + gets(prob, T, 6) * u[T[3]]
@inbounds @muladd getγ(prob, T, u) = gets(prob, T, 7) * u[T[1]] + gets(prob, T, 8) * u[T[2]] + gets(prob, T, 9) * u[T[3]]

function par_fvm_eqs_edge!(du::AbstractVector{V}, t, (vj, j), (vjnb, jnb), α, β, γ, prob, T) where {V}
    x, y = get_control_volume_edge_midpoints(prob, T, j)
    xn, yn = get_normals(prob, T, j)
    ℓ = get_lengths(prob, T, j)
    q1, q2 = get_flux(prob, x, y, t, α, β, γ)
    summand = -(q1 * xn + q2 * yn) * ℓ
    if is_interior_or_neumann_node(prob, vj)
        lock(fvm_splock) do
            @inbounds du[vj] += summand
        end
    end
    if is_interior_or_neumann_node(prob, vjnb)
        lock(fvm_splock) do
            @inbounds du[vjnb] -= summand
        end
    end
    return nothing
end
@inline function par_fvm_eqs_edge!(du, t, α, β, γ, prob, T)
    i, j, k = indices(T)
    par_fvm_eqs_edge!(du, t, (i, 1), (j, 2), α, β, γ, prob, T)#unrolled
    par_fvm_eqs_edge!(du, t, (j, 2), (k, 3), α, β, γ, prob, T)
    par_fvm_eqs_edge!(du, t, (k, 3), (i, 1), α, β, γ, prob, T)
    return nothing
end

@inline function par_fvm_eqs_interior_element!(du, u, t, prob, T)
    α = getα(prob, T, u)
    β = getβ(prob, T, u)
    γ = getγ(prob, T, u)
    par_fvm_eqs_edge!(du, t, α, β, γ, prob, T)
    return nothing
end
@inline function par_fvm_eqs_interior_elements!(du, u, t, prob, interior_elements)
    Threads.@threads for V in interior_elements
        par_fvm_eqs_interior_element!(du, u, t, prob, V)
    end
    return nothing
end

@inline function par_fvm_eqs_boundary_element!(du, u, t, prob, T)
    α = getα(prob, T, u)
    β = getβ(prob, T, u)
    γ = getγ(prob, T, u)
    interior_edges = get_interior_edges(prob, T)
    for ((vj, j), (vjnb, jnb)) in interior_edges
        par_fvm_eqs_edge!(du, t, (vj, j), (vjnb, jnb), α, β, γ, prob, T)
    end
    return nothing
end
@inline function par_fvm_eqs_boundary_elements!(du, u, t, prob, boundary_elements)
    Threads.@threads for V in boundary_elements
        par_fvm_eqs_boundary_element!(du, u, t, prob, V)
    end
    return nothing
end

@inline function par_fvm_eqs_source_contribution!(du, u, t, j, prob)
    x, y = get_point(prob, j)
    V = get_volumes(prob, j)
    @inbounds R = get_reaction(prob, x, y, t, u[j])
    lock(fvm_splock) do
        @inbounds @muladd du[j] = du[j] / V + R
    end
    return nothing
end
@inline function par_fvm_eqs_source_contributions!(du, u, t, prob, interior_or_neumann_nodes)
    @floop for j in interior_or_neumann_nodes
        par_fvm_eqs_source_contribution!(du, u, t, j, prob)
    end
    return nothing
end

@inline function par_update_dudt_nodes!(du, u, t, prob, dudt_nodes)
    @floop for j in dudt_nodes
        evaluate_boundary_function!(du, u, t, j, prob)
    end
    return nothing
end

"""
    fvm_eqs!(du::AbstractVector{T}, u, p, t) where {T}

Evaluates the system of ODEs from the finite volume approximation, where 
the parameters `p` are `(prob, flux_cache, shape_coeffs)`, where `prob` is the 
corresponding [`FVMProblem`](@ref), `flux_cache` is a cache for storing the flux vector, 
and `shape_coeffs` is a cache for storing the shape function coefficients `(α, β, γ)`.
"""
function par_fvm_eqs!(du::AbstractVector{T}, u, p, t) where {T}
    prob,
    collected_dudt_nodes,
    collected_interior_or_neumann_nodes,
    collected_boundary_elements,
    collected_interior_elements,
    collected_elements,
    collected_dirichlet_nodes = p
    fill!(du, zero(T))
    par_fvm_eqs_interior_elements!(du, u, t, prob, collected_interior_elements)
    par_fvm_eqs_boundary_elements!(du, u, t, prob, collected_boundary_elements)
    par_fvm_eqs_source_contributions!(du, u, t, prob, collected_interior_or_neumann_nodes)
    par_update_dudt_nodes!(du, u, t, prob, collected_dudt_nodes)
    return nothing
end

function par_update_dirichlet_nodes!(u, t, prob, dirichlet_nodes)
    @inbounds @floop for j in dirichlet_nodes
        u[j] = evaluate_boundary_function(u, t, j, prob)
    end
    return nothing
end
function par_update_dirichlet_nodes!(integrator)
    @inbounds par_update_dirichlet_nodes!(integrator.u, integrator.t, integrator.p[1], integrator.p[end])
    return nothing
end