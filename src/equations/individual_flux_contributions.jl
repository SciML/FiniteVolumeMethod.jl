# primitive: flux contribution without a boundary condition
@inline function _non_neumann_get_flux(prob, x, y, t, α::T, β, γ, nx, ny) where {T}
    qx, qy = eval_flux_function(prob, x, y, t, α, β, γ)
    qn = qx * nx + qy * ny
    return qn * one(eltype(T))
end

# primitive: flux contribution with a Neumann boundary condition
@inline function _neumann_get_flux(prob, x, y, t, u::T, i, j) where {T}
    function_index = get_neumann_fidx(prob, i, j)
    qn = eval_condition_fnc(prob, function_index, x, y, t, u) * one(eltype(T))
    return qn * one(eltype(T))
end

# primitive: get flux contribution without a boundary condition. This is used as a function barrier
@inline function _get_flux(prob::AbstractFVMProblem, x, y, t, α::A, β, γ, nx, ny) where {A}
    qn = _non_neumann_get_flux(prob, x, y, t, α, β, γ, nx, ny) * one(eltype(A))
    return qn
end

# get flux contribution for a non-system, also picking up the cv components first
@inline function get_flux(prob::AbstractFVMProblem, props, α::A, β, γ, t::T, edge_index) where {A,T}
    x, y, nx, ny, ℓ = _get_cv_components(props, edge_index)
    qn = _get_flux(prob, x, y, t, α, β, γ, nx, ny)
    return qn * ℓ
end

# primitive: get flux contribution for a system without a boundary condition for a single variable. This is used as a function barrier
@inline function _get_flux(prob::FVMSystem, x, y, t, α::T, β, γ, nx, ny, var) where {T}
    qn = _get_flux(get_equation(prob, var), x, y, t, α, β, γ, nx, ny) * one(eltype(T))
    return qn
end

# get flux contribution for a system, also picking up the cv components first, and getting it for all variables
function get_flux(prob::FVMSystem, props, α::A, β, γ, t::T, edge_index) where {A,T}
    x, y, nx, ny, ℓ = _get_cv_components(props, edge_index)
    qn = ntuple(_neqs(prob)) do var
        _qn = _get_flux(prob, x, y, t, α, β, γ, nx, ny, var)
        return _qn * ℓ
    end
    return qn
end