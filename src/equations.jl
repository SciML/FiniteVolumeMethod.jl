function idx_is_point_condition(prob::FVMProblem, idx)
    return idx ∈ keys(prob.conditions.point_conditions)
end
function idx_is_point_condition(prob::FVMSystem{N}, idx, var) where {N}
    return idx_is_point_condition(prob.problens[var], idx)
end
function idx_is_dudt_condition(prob::FVMProblem, idx)
    return prob.conditions.point_conditions[idx][1] == Dudt
end
function idx_is_dudt_condition(prob::FVMSystem{N}, idx, var) where {N}
    return idx_is_dudt_condition(prob.problems[var], idx)
end
function idx_is_neumann_condition(prob::FVMProblem, i, j)
    return (i, j) ∈ keys(prob.conditions.edge_conditions)
end
function idx_is_neumann_condition(prob::FVMSystem{N}, i, j, var) where {N}
    return idx_is_neumann_condition(prob.problems[var], i, j)
end

function get_shape_function_coefficients(props::TriangleProperties, T, u, ::FVMProblem)
    i, j, k = indices(T)
    s₁, s₂, s₃, s₄, s₅, s₆, s₇, s₈, s₉ = props.shape_function_coefficients
    α = s₁ * u[i] + s₂ * u[j] + s₃ * u[k]
    β = s₄ * u[i] + s₅ * u[j] + s₆ * u[k]
    γ = s₇ * u[i] + s₈ * u[j] + s₉ * u[k]
    return α, β, γ
end
function get_shape_function_coefficients(props::TriangleProperties, T, u, ::FVMSystem{N}) where {N}
    i, j, k = indices(T)
    s₁, s₂, s₃, s₄, s₅, s₆, s₇, s₈, s₉ = props.shape_function_coefficients
    α = ntuple(ℓ -> s₁ * u[ℓ, i] + s₂ * u[ℓ, j] + s₃ * u[ℓ, k], N)
    β = ntuple(ℓ -> s₄ * u[ℓ, i] + s₅ * u[ℓ, j] + s₆ * u[ℓ, k], N)
    γ = ntuple(ℓ -> s₇ * u[ℓ, i] + s₈ * u[ℓ, j] + s₉ * u[ℓ, k], N)
    return α, β, γ
end

function get_flux(prob::FVMProblem, props, α, β, γ, t, i, j, edge_index)
    # For checking if an edge is Neumann, we need only check e.g. (i, j) and not (j, i), since we do not allow for internal Neumann edges.
    ij_is_neumann = idx_is_neumann_condition(prob, i, j)
    x, y = props.control_volume_midpoints[edge_index]
    nx, ny = props.control_volume_normals[edge_index]
    ℓ = props.control_volume_edge_lengths[edge_index]
    if !ij_is_neumann
        qx, qy = prob.flux_function(x, y, t, α, β, γ, prob.flux_parameters)
        qn = qx * nx + qy * ny
    else
        function_index = prob.conditions.edge_conditions[(i, j)]
        a, ap = prob.conditions.functions[function_index], prob.conditions.parameters[function_index]
        qn = a(x, y, t, α * x + β * y + γ, ap)
    end
    return qn * ℓ
end
function get_flux(prob::FVMSystem{N}, props, α, β, γ, t, i, j, edge_index) where {N}
    x, y = props.control_volume_midpoints[edge_index]
    nx, ny = props.control_volume_normals[edge_index]
    ℓ = props.control_volume_edge_lengths[edge_index]
    u_shape = ntuple(ℓ -> α[ℓ] * x + β[ℓ] * y + γ[ℓ], N)
    qn = ntuple(N) do ℓ
        ij_is_neumann = idx_is_neumann_condition(prob, i, j, ℓ)
        if !ij_is_neumann
            qx, qy = prob.problems[ℓ].flux_function(x, y, t, α, β, γ, prob.problems[ℓ].flux_parameters)
            qn = qx * nx + qy * ny
        else
            function_index = prob.problems[ℓ].conditions.edge_conditions[(i, j)]
            a, ap = prob.problems[ℓ].conditions.functions[function_index], prob.problems[ℓ].conditions.parameters[function_index]
            qn = a(x, y, t, u_shape, ap)
        end
        return qn * ℓ
    end
    return qn
end

function get_fluxes(prob, props, α, β, γ, t)
    q1 = get_flux(prob, props, α, β, γ, t, i, j, 1)
    q2 = get_flux(prob, props, α, β, γ, t, j, k, 2)
    q3 = get_flux(prob, props, α, β, γ, t, k, i, 3)
    return q1, q2, q3
end

function update_du!(du, prob::FVMProblem, i, j, k, summand₁, summand₂, summand₃)
    i_is_pc = idx_is_point_condition(prob, i)
    j_is_pc = idx_is_point_condition(prob, j)
    k_is_pc = idx_is_point_condition(prob, k)
    if !i_is_pc
        du[i] -= summand₁
        du[j] += summand₁
    end
    if !j_is_pc
        du[j] -= summand₂
        du[k] += summand₂
    end
    if !k_is_pc
        du[k] -= summand₃
        du[i] += summand₃
    end
end
function update_du!(du, prob::FVMSystem{N}, i, j, k, ℓ, summand₁, summand₂, summand₃) where {N}
    i_is_pc = idx_is_point_condition(prob, i, ℓ)
    j_is_pc = idx_is_point_condition(prob, j, ℓ)
    k_is_pc = idx_is_point_condition(prob, k, ℓ)
    @. begin
        if !i_is_pc
            du[:, i] -= summand₁
            du[:, j] += summand₁
        end
        if !j_is_pc
            du[:, j] -= summand₂
            du[:, k] += summand₂
        end
        if !k_is_pc
            du[:, k] -= summand₃
            du[:, i] += summand₃
        end
    end
end

function fvm_eqs_single_triangle!(du, u, prob, t, T)
    i, j, k = indices(T)
    props = prob.mesh.triangle_props[(i, j, k)]
    α, β, γ = get_shape_function_coefficients(props, T, u, prob)
    summand₁, summand₂, summand₃ = get_fluxes(prob, props, α, β, γ, t)
    update_du!(du, prob, i, j, k, summand₁, summand₂, summand₃)
    return nothing
end

function fvm_eqs_single_source_contribution!(du, u, prob::FVMProblem, t, i)
    p = get_point(prob.mesh.triangulation, i)
    x, y = getxy(p)
    if !idx_is_point_condition(prob, i)
        du[i] += prob.source_function(x, y, t, u[i], prob.source_parameters)
    elseif idx_is_dudt_condition(prob, i)
        function_index = prob.conditions.point_conditions[i][2]
        a, ap = prob.conditions.functions[function_index], prob.conditions.parameters[function_index]
        du[i] = a(x, y, t, u[i], ap)
    else # Dirichlet
        du[i] = zero(eltype(du))
    end
    return nothing
end
function fvm_eqs_single_source_contribution!(du, u, prob::FVMSystem{N}, t, i) where {N}
    p = get_point(prob.mesh.triangulation, i)
    x, y = getxy(p)
    uᵢ = @views u[:, i]
    for j in 1:N
        if !idx_is_point_condition(prob, i, j)
            du[j, i] += prob.problems[j].source_function(x, y, t, uᵢ, prob.problems[j].source_parameters)
        elseif idx_is_dudt_condition(prob, i, j)
            function_index = prob.problems[j].conditions.point_conditions[i][2]
            a, ap = prob.problems[j].conditions.functions[function_index], prob.problems[j].conditions.parameters[function_index]
            du[j, i] = a(x, y, t, uᵢ, ap)
        else # Dirichlet
            du[j, i] = zero(eltype(du))
        end
    end
end

function serial_fvm_eqs!(du, u, prob, t)
    fill!(du, zero(eltype(du)))
    for T in each_solid_triangle(prob.mesh.triangulation)
        fvm_eqs_single_triangle!(du, u, prob, t, T)
    end
    for i in each_solid_vertex(prob.mesh.triangulation)
        fvm_eqs_single_source_contribution!(du, u, prob, t, i)
    end
    return nothing
end

function combine_duplicated_du!(du, duplicated_du, prob)
    if prob isa FVMProblem
        for _du in eachcol(duplicated_du)
            du .+= _du
        end
    else
        for i in axes(duplicated_du, 3)
            du .+= duplicated_du[:, :, i]
        end
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
    Threads.@threads for (triangle_range, chunk_idx) in chunked_solid_triangles
        for triangle_idx in triangle_range
            T = solid_triangles[triangle_idx]
            if prob isa FVMProblem
                @views fvm_eqs_single_triangle!(_duplicated_du[:, chunk_idx], u, prob, t, T)
            else
                @views fvm_eqs_single_triangle!(_duplicated_du[:, :, chunk_idx], u, prob, t, T)
            end
        end
    end
    combine_duplicated_du!(du, duplicated_du, prob)
    Threads.@threads for i in solid_vertices
        fvm_eqs_single_source_contribution!(du, u, prob, t, i)
    end
    return nothing
end

function fvm_eqs!(du, u, p, t)
    prob, parallel = p.prob, p.parallel
    if parallel == Val(false)
        return serial_fvm_eqs!(du, u, prob, t)
    else
        return parallel_fvm_eqs!(du, u, p, t)
    end
end

function update_dirichlet_nodes_single!(u, t, prob::FVMProblem, i, condition, function_index)
    if condition == Dirichlet
        a, ap = prob.conditions.functions[function_index], prob.conditions.parameters[function_index]
        p = get_point(prob.mesh.triangulation, i)
        x, y = getxy(p)
        u[i] = a(x, y, t, u[i], ap)
    end
    return nothing
end
function update_dirichlet_nodes_single!(u, t, prob::FVMSystem{N}, i, j, condition, function_index) where {N}
    if condition == Dirichlet
        a, ap = prob.problems[j].conditions.functions[function_index], prob.problems[j].conditions.parameters[function_index]
        p = get_point(prob.mesh.triangulation, i)
        x, y = getxy(p)
        @views u[j, i] = a(x, y, t, u[:, i], ap)
    end
end

function serial_update_dirichlet_nodes!(u, t, prob::FVMProblem)
    for (i, (condition, function_index)) in prob.conditions.point_conditions
        update_dirichlet_nodes_single!(u, t, prob, i, condition, function_index)
    end
    return nothing
end
function serial_update_dirichlet_nodes!(u, t, prob::FVMSystem{N}) where {N}
    for j in 1:N
        for (i, (condition, function_index)) in prob.problems[j].conditions.point_conditions
            update_dirichlet_nodes_single!(u, t, prob, i, j, condition, function_index)
        end
    end
end

function parallel_update_dirichlet_nodes!(u, t, p, prob::FVMProblem)
    point_conditions = p.point_conditions
    Threads.@threads for i in point_conditions
        point_conditions_dict = prob.conditions.point_conditions
        condition, function_index = point_conditions_dict[i]
        update_dirichlet_nodes_single!(u, t, prob, i, condition, function_index)
    end
    return nothing
end
function parallel_update_dirichlet_nodes!(u, t, p, prob::FVMSystem{N}) where {N}
    point_conditions = p.point_conditions
    for j in 1:N
        Threads.@threads for i in point_conditions[j]
            point_conditions_dict = prob.problems[j].conditions.point_conditions
            condition, function_index = point_conditions_dict[i]
            update_dirichlet_nodes_single!(u, t, prob, i, condition, function_index)
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