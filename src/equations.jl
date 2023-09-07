@inline function get_shape_function_coefficients(props::TriangleProperties, T, u, ::AbstractFVMProblem)
    i, j, k = indices(T)
    s₁, s₂, s₃, s₄, s₅, s₆, s₇, s₈, s₉ = props.shape_function_coefficients
    α = s₁ * u[i] + s₂ * u[j] + s₃ * u[k]
    β = s₄ * u[i] + s₅ * u[j] + s₆ * u[k]
    γ = s₇ * u[i] + s₈ * u[j] + s₉ * u[k]
    return α, β, γ
end
@inline function get_shape_function_coefficients(props::TriangleProperties, T, u, ::FVMSystem{N}) where {N}
    i, j, k = indices(T)
    s₁, s₂, s₃, s₄, s₅, s₆, s₇, s₈, s₉ = props.shape_function_coefficients
    α = ntuple(ℓ -> s₁ * u[ℓ, i] + s₂ * u[ℓ, j] + s₃ * u[ℓ, k], Val(N))
    β = ntuple(ℓ -> s₄ * u[ℓ, i] + s₅ * u[ℓ, j] + s₆ * u[ℓ, k], Val(N))
    γ = ntuple(ℓ -> s₇ * u[ℓ, i] + s₈ * u[ℓ, j] + s₉ * u[ℓ, k], Val(N))
    return α, β, γ
end

@inline function _get_cv_components(props, edge_index)
    x, y = props.cv_edge_midpoints[edge_index]
    nx, ny = props.cv_edge_normals[edge_index]
    ℓ = props.cv_edge_lengths[edge_index]
    return x, y, nx, ny, ℓ
end
@inline function _non_neumann_get_flux(prob, x, y, t, α::T, β, γ, nx, ny) where {T}
    qx, qy = eval_flux_function(prob, x, y, t, α, β, γ)
    qn = qx * nx + qy * ny
    return qn * one(eltype(T))
end
@inline function _neumann_get_flux(prob, x, y, t, u::T, i, j) where {T}
    function_index = get_neumann_fidx(prob, i, j)
    qn = eval_condition_fnc(prob, function_index, x, y, t, u) * one(T)
    return qn::T
end
@inline function _get_flux(prob, x, y, t, α, β, γ, nx, ny, i, j, u::U) where {U}
    # For checking if an edge is Neumann, we need only check e.g. (i, j) and not (j, i), since we do not allow for internal Neumann edges.
    ij_is_neumann = is_neumann_edge(prob, i, j)
    qn = if !ij_is_neumann
        _non_neumann_get_flux(prob, x, y, t, α, β, γ, nx, ny)
    else
        _neumann_get_flux(prob, x, y, t, u, i, j)
    end
    return qn
end
@inline function get_flux(prob::AbstractFVMProblem, props, α::A, β, γ, t::T, i, j, edge_index) where {A,T}
    x, y, nx, ny, ℓ = _get_cv_components(props, edge_index)
    qn = _get_flux(prob, x, y, t, α, β, γ, nx, ny, i, j, α * x + β * y + γ)
    return qn * ℓ
end
@inline function _get_flux(prob, x, y, t, α::T, β, γ, u_shape, nx, ny, i, j, var) where {T}
    qn = _get_flux(get_equation(prob, var), x, y, t, α, β, γ, nx, ny, i, j, u_shape) * one(eltype(T))
    return qn
end
function get_flux(prob::FVMSystem{N}, props, α::A, β, γ, t::T, i, j, edge_index) where {N,A,T}
    x, y, nx, ny, ℓ = _get_cv_components(props, edge_index)
    u_shape = ntuple(var -> α[var] * x + β[var] * y + γ[var], Val(N))
    qn = ntuple(Val(N)) do var
        qn = _get_flux(prob, x, y, t, α, β, γ, u_shape, nx, ny, i, j, var)
        return qn * ℓ
    end
    return qn
end

@inline function get_fluxes(prob, props, α, β, γ, i, j, k, t)
    q1 = get_flux(prob, props, α, β, γ, t, i, j, 1)
    q2 = get_flux(prob, props, α, β, γ, t, j, k, 2)
    q3 = get_flux(prob, props, α, β, γ, t, k, i, 3)
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
    summand₁, summand₂, summand₃ = get_fluxes(prob, props, α, β, γ, i, j, k, t)
    update_du!(du, prob, i, j, k, summand₁, summand₂, summand₃)
    return nothing
end

@inline function get_source_contribution(prob::AbstractFVMProblem, u::T, t, i) where {T}
    p = get_point(prob, i)
    x, y = getxy(p)
    if !has_condition(prob, i)
        S = eval_source_fnc(prob, x, y, t, u[i]) * one(eltype(T))
    elseif is_dirichlet_node(prob, i)
        S = zero(eltype(T))
    else # Dudt
        function_index = get_dudt_fidx(prob, i)
        S = eval_condition_fnc(prob, function_index, x, y, t, u[i]) * one(eltype(T))
    end
    return S::eltype(T)
end
@inline function get_source_contribution(prob::FVMSystem{N}, u::T, t, i, var) where {T,N}
    p = get_point(prob, i)
    x, y = getxy(p)
    if !has_condition(prob, i, var)
        S = @views eval_source_fnc(prob, var, x, y, t, u[:, i]) * one(eltype(T))
    elseif is_dirichlet_node(prob, i)
        S = zero(eltype(T))
    else # Dudt
        function_index = get_dudt_fidx(prob, i, var)
        S = @views eval_condition_fnc(prob, function_index, var, x, y, t, u[:, i]) * one(eltype(T))
    end
    return S::eltype(T)
end

@inline function fvm_eqs_single_source_contribution!(du::T, u, prob::AbstractFVMProblem, t, i) where {T}
    S = get_source_contribution(prob, u, t, i)::eltype(T)
    if !has_condition(prob, i)
        du[i] = du[i] / get_volume(prob, i) + S
    else
        du[i] = S
    end
    return nothing
end
@inline function fvm_eqs_single_source_contribution!(du::T, u, prob::FVMSystem{N}, t, i) where {N,T}
    for var in 1:N
        S = get_source_contribution(prob, u, t, i, var)::eltype(T)
        if !has_condition(prob, i, var)
            du[var, i] = du[var, i] / get_volume(prob, i) + S
        else
            du[var, i] = S
        end
    end
    return nothing
end

function get_triangle_contributions!(du, u, prob, t)
    for T in each_solid_triangle(prob.mesh.triangulation)
        fvm_eqs_single_triangle!(du, u, prob, t, T)
    end
    return nothing
end
function get_source_contributions!(du, u, prob, t)
    for i in each_solid_vertex(prob.mesh.triangulation)
        fvm_eqs_single_source_contribution!(du, u, prob, t, i)
    end
    return nothing
end

function serial_fvm_eqs!(du, u, prob, t)
    fill!(du, zero(eltype(du)))
    get_triangle_contributions!(du, u, prob, t)
    get_source_contributions!(du, u, prob, t)
    return du
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
function combine_duplicated_du!(du, duplicated_du, prob)
    if prob isa FVMSystem
        for i in axes(duplicated_du, 3)
            du .+= duplicated_du[:, :, i]
        end
    else
        for _du in eachcol(duplicated_du)
            du .+= _du
        end
    end
    return nothing
end
function get_parallel_source_contributions!(du, u, prob, t, solid_vertices)
    Threads.@threads for i in solid_vertices
        fvm_eqs_single_source_contribution!(du, u, prob, t, i)
    end
    return nothing
end

function parallel_fvm_eqs!(du, u, p, t)
    duplicated_du, solid_triangles,
    solid_vertices, chunked_solid_triangles,
    prob = p.duplicated_du, p.solid_triangles,
    p.solid_vertices, p.chunked_solid_triangles,
    p.prob
    fill!(du, zero(eltype(du)))
    _duplicated_du = get_tmp(duplicated_du, du)
    fill!(_duplicated_du, zero(eltype(du)))
    get_parallel_triangle_contributions!(_duplicated_du, u, prob, t, chunked_solid_triangles, solid_triangles)
    combine_duplicated_du!(du, _duplicated_du, prob)
    get_parallel_source_contributions!(du, u, prob, t, solid_vertices)
    return du
end

function fvm_eqs!(du, u, p, t)
    prob, parallel = p.prob, p.parallel
    if parallel == Val(false)
        return serial_fvm_eqs!(du, u, prob, t)
    else
        return parallel_fvm_eqs!(du, u, p, t)
    end
end

@inline function get_dirichlet_condition(prob::AbstractFVMProblem, u::T, t, i, function_index) where {T}
    p = get_point(prob, i)
    x, y = getxy(p)
    return eval_condition_fnc(prob, function_index, x, y, t, u[i])::eltype(T)
end
@inline function get_dirichlet_condition(prob::FVMSystem{N}, u::T, t, i, var, function_index) where {N,T}
    p = get_point(prob, i)
    x, y = getxy(p)
    return @views eval_condition_fnc(prob, function_index, var, x, y, t, u[:, i])::eltype(T)
end

@inline function update_dirichlet_nodes_single!(u::T, t, prob::AbstractFVMProblem, i, function_index) where {T}
    d = get_dirichlet_condition(prob, u, t, i, function_index)::eltype(T)
    u[i] = d
    return nothing
end
@inline function update_dirichlet_nodes_single!(u::T, t, prob::FVMSystem{N}, i, var, function_index) where {N,T}
    d = get_dirichlet_condition(prob, u, t, i, var, function_index)::eltype(T)
    u[var, i] = d
    return nothing
end

function serial_update_dirichlet_nodes!(u, t, prob::AbstractFVMProblem)
    for (i, function_index) in get_dirichlet_nodes(prob)
        update_dirichlet_nodes_single!(u, t, prob, i, function_index)
    end
    return nothing
end
function serial_update_dirichlet_nodes!(u, t, prob::FVMSystem{N}) where {N}
    for var in 1:N
        for (i, function_index) in get_dirichlet_nodes(prob, var)
            update_dirichlet_nodes_single!(u, t, prob, i, var, function_index)
        end
    end
end

function parallel_update_dirichlet_nodes!(u, t, p, prob::AbstractFVMProblem)
    dirichlet_nodes = p.dirichlet_nodes
    Threads.@threads for i in dirichlet_nodes
        function_index = get_dirichlet_fidx(prob, i)
        update_dirichlet_nodes_single!(u, t, prob, i, function_index)
    end
    return nothing
end
function parallel_update_dirichlet_nodes!(u, t, p, prob::FVMSystem{N}) where {N}
    dirichlet_nodes = p.dirichlet_nodes
    for var in 1:N
        Threads.@threads for i in dirichlet_nodes[j]
            function_index = get_dirichlet_fidx(prob, i, var)
            update_dirichlet_nodes_single!(u, t, prob, i, var, function_index)
        end
    end
    return nothing
end

function update_dirichlet_nodes!(integrator)
    prob, parallel = integrator.p.prob, integrator.p.parallel
    if parallel == Val(false)
        return serial_update_dirichlet_nodes!(integrator.u, integrator.t, prob)
    else
        return parallel_update_dirichlet_nodes!(integrator.u, integrator.t, integrator.p, prob)
    end
    return nothing
end