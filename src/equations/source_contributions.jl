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

function get_source_contributions!(du, u, prob, t)
    for i in each_solid_vertex(prob.mesh.triangulation)
        fvm_eqs_single_source_contribution!(du, u, prob, t, i)
    end
    return nothing
end

function get_parallel_source_contributions!(du, u, prob, t, solid_vertices)
    Threads.@threads for i in solid_vertices
        fvm_eqs_single_source_contribution!(du, u, prob, t, i)
    end
    return nothing
end
