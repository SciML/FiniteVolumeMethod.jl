function has_condition(prob::AbstractFVMProblem, node)
    return is_dudt_node(prob, node) || is_dirichlet_node(prob, node)
end
has_condition(prob::FVMSystem, node, var) = has_condition(prob.problems[var], node)
is_dudt_node(prob::AbstractFVMProblem, node) = node ∈ keys(prob.conditions.dudt_nodes)
is_dudt_node(prob::FVMSystem, node, var) = is_dudt_node(prob.problems[var], node)
is_dirichlet_node(prob::AbstractFVMProblem, node) = node ∈ keys(prob.conditions.dirichlet_nodes)
is_dirichlet_node(prob::FVMSystem, node, var) = is_dirichlet_node(prob.problems[var], node)
is_neumann_edge(prob::AbstractFVMProblem, i, j) = (i, j) ∈ keys(prob.conditions.neumann_edges)
is_neumann_edge(prob::FVMSystem, i, j, var) = is_neumann_edge(prob.problems[var], i, j)
is_constrained_edge(prob::AbstractFVMProblem, i, j) = (i, j) ∈ keys(prob.conditions.constrained_edges)
is_constrained_edge(prob::FVMSystem, i, j, var) = is_constrained_edge(prob.problems[var], i, j)

function get_shape_function_coefficients(props::TriangleProperties, T, u, ::AbstractFVMProblem)
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
    α = ntuple(ℓ -> s₁ * u[ℓ, i] + s₂ * u[ℓ, j] + s₃ * u[ℓ, k], Val(N))
    β = ntuple(ℓ -> s₄ * u[ℓ, i] + s₅ * u[ℓ, j] + s₆ * u[ℓ, k], Val(N))
    γ = ntuple(ℓ -> s₇ * u[ℓ, i] + s₈ * u[ℓ, j] + s₉ * u[ℓ, k], Val(N))
    return α, β, γ
end

function get_flux(prob::AbstractFVMProblem, props, α, β, γ, t, i, j, edge_index)
    # For checking if an edge is Neumann, we need only check e.g. (i, j) and not (j, i), since we do not allow for internal Neumann edges.
    ij_is_neumann = is_neumann_edge(prob, i, j)
    x, y = props.cv_edge_midpoints[edge_index]
    nx, ny = props.cv_edge_normals[edge_index]
    ℓ = props.cv_edge_lengths[edge_index]
    if !ij_is_neumann
        qx, qy = eval_flux_function(prob, x, y, t, α, β, γ)
        qn = qx * nx + qy * ny
    else
        function_index = prob.conditions.neumann_edges[(i, j)]
        a, ap = prob.conditions.functions[function_index], prob.conditions.parameters[function_index]
        qn = a(x, y, t, α * x + β * y + γ, ap)
    end
    return qn * ℓ
end
function get_flux(prob::FVMSystem{N}, props, α, β, γ, t, i, j, edge_index) where {N}
    x, y = props.cv_edge_midpoints[edge_index]
    nx, ny = props.cv_edge_normals[edge_index]
    ℓ = props.cv_edge_lengths[edge_index]
    u_shape = ntuple(var -> α[var] * x + β[var] * y + γ[var], Val(N))
    qn = ntuple(Val(N)) do var
        ij_is_neumann = is_neumann_edge(prob, i, j, ℓ)
        if !ij_is_neumann
            qx, qy = eval_flux_function(prob.problems[var], x, y, t, α[var], β[var], γ[var])
            qn = qx * nx + qy * ny
        else
            function_index = prob.problems[var].conditions.neumann_edges[(i, j)]
            a, ap = prob.problems[var].conditions.functions[function_index], prob.problems[var].conditions.parameters[function_index]
            qn = a(x, y, t, u_shape, ap)
        end
        return qn * ℓ
    end
    return qn
end

function get_fluxes(prob, props, α, β, γ, i, j, k, t)
    q1 = get_flux(prob, props, α, β, γ, t, i, j, 1)
    q2 = get_flux(prob, props, α, β, γ, t, j, k, 2)
    q3 = get_flux(prob, props, α, β, γ, t, k, i, 3)
    return q1, q2, q3
end

function update_du!(du, ::AbstractFVMProblem, i, j, k, summand₁, summand₂, summand₃)
    du[i] = du[i] + summand₃ - summand₁
    du[j] = du[j] + summand₁ - summand₂
    du[k] = du[k] + summand₂ - summand₃
    return nothing
end
function update_du!(du, ::FVMSystem{N}, i, j, k, ℓ, summand₁, summand₂, summand₃) where {N}
    for var in 1:N
        du[var, i] = du[var, i] + summand₃[var] - summand₁[var]
        du[var, j] = du[var, j] + summand₁[var] - summand₂[var]
        du[var, k] = du[var, k] + summand₂[var] - summand₃[var]
    end
    return nothing
end

function fvm_eqs_single_triangle!(du, u, prob, t, T)
    i, j, k = indices(T)
    props = prob.mesh.triangle_props[(i, j, k)]
    α, β, γ = get_shape_function_coefficients(props, T, u, prob)
    summand₁, summand₂, summand₃ = get_fluxes(prob, props, α, β, γ, i, j, k, t)
    update_du!(du, prob, i, j, k, summand₁, summand₂, summand₃)
    return nothing
end

function fvm_eqs_single_source_contribution!(du, u, prob::AbstractFVMProblem, t, i)
    p = get_point(prob.mesh.triangulation, i)
    x, y = getxy(p)
    if !has_condition(prob, i)
        du[i] = du[i] / prob.mesh.cv_volumes[i] + prob.source_function(x, y, t, u[i], prob.source_parameters)
    elseif is_dirichlet_node(prob, i)
        du[i] = zero(eltype(du))
    else # Dudt
        function_index = prob.conditions.dudt_nodes[i]
        a, ap = prob.conditions.functions[function_index], prob.conditions.parameters[function_index]
        du[i] = a(x, y, t, u[i], ap)
    end
    return nothing
end
function fvm_eqs_single_source_contribution!(du, u, prob::FVMSystem{N}, t, i) where {N}
    p = get_point(prob.mesh.triangulation, i)
    x, y = getxy(p)
    uᵢ = @views u[:, i]
    for j in 1:N
        if !has_condition(prob, i, j)
            du[j, i] = du[j, i] / prob.mesh.cv_volumes[i] + prob.problems[j].source_function(x, y, t, uᵢ, prob.problems[j].source_parameters)
        elseif is_dirichlet_node(prob, i, j)
            # Need to do Dirichlet first in case Dirichlet and Dudt overlap at this node - Dirichlet takes precedence
            du[j, i] = zero(eltype(du))
        else # Dudt
            function_index = prob.problems[j].conditions.dudt_nodes[i]
            a, ap = prob.problems[j].conditions.functions[function_index], prob.problems[j].conditions.parameters[function_index]
            du[j, i] = a(x, y, t, uᵢ, ap)
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
    return du
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
            if prob isa FVMSystem
                @views fvm_eqs_single_triangle!(_duplicated_du[:, :, chunk_idx], u, prob, t, T)
            else
                @views fvm_eqs_single_triangle!(_duplicated_du[:, chunk_idx], u, prob, t, T)
            end
        end
    end
    combine_duplicated_du!(du, _duplicated_du, prob)
    Threads.@threads for i in solid_vertices
        fvm_eqs_single_source_contribution!(du, u, prob, t, i)
    end
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

function update_dirichlet_nodes_single!(u, t, prob::AbstractFVMProblem, i, function_index)
    a, ap = prob.conditions.functions[function_index], prob.conditions.parameters[function_index]
    p = get_point(prob.mesh.triangulation, i)
    x, y = getxy(p)
    u[i] = a(x, y, t, u[i], ap)
    return nothing
end
function update_dirichlet_nodes_single!(u, t, prob::FVMSystem{N}, i, var, function_index) where {N}
    a, ap = prob.problems[var].conditions.functions[function_index], prob.problems[var].conditions.parameters[function_index]
    p = get_point(prob.mesh.triangulation, i)
    x, y = getxy(p)
    @views u[j, i] = a(x, y, t, u[:, i], ap)
end

function serial_update_dirichlet_nodes!(u, t, prob::AbstractFVMProblem)
    for (i, function_index) in prob.conditions.dirichlet_nodes
        update_dirichlet_nodes_single!(u, t, prob, i, function_index)
    end
    return nothing
end
function serial_update_dirichlet_nodes!(u, t, prob::FVMSystem{N}) where {N}
    for j in 1:N
        for (i, function_index) in prob.problems[j].conditions.dirichlet_nodes
            update_dirichlet_nodes_single!(u, t, prob, i, j, function_index)
        end
    end
end

function parallel_update_dirichlet_nodes!(u, t, p, prob::AbstractFVMProblem)
    dirichlet_nodes = p.dirichlet_nodes
    Threads.@threads for i in dirichlet_nodes
        function_index = prob.conditions.dirichlet_nodes[i]
        update_dirichlet_nodes_single!(u, t, prob, i, function_index)
    end
    return nothing
end
function parallel_update_dirichlet_nodes!(u, t, p, prob::FVMSystem{N}) where {N}
    dirichlet_nodes = p.dirichlet_nodes
    for var in 1:N
        Threads.@threads for i in dirichlet_nodes[j]
            function_index = prob.problems[j].conditions.dirichlet_nodes[i]
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