# primitive: get dirichlet value for a non-system
@inline function get_dirichlet_condition(prob::AbstractFVMProblem, u::T, t, i, function_index) where {T}
    p = get_point(prob, i)
    x, y = getxy(p)
    return eval_condition_fnc(prob, function_index, x, y, t, u[i]) 
end

# primitive: get dirichlet value for a system
@inline function get_dirichlet_condition(prob::FVMSystem, u::T, t, i, var, function_index) where {T}
    p = get_point(prob, i)
    x, y = getxy(p)
    return @views eval_condition_fnc(prob, function_index, var, x, y, t, u[:, i]) 
end

# get the dirichlet value and update u non-system. need this function barriers for inference
@inline function update_dirichlet_nodes_single!(u::T, t, prob::AbstractFVMProblem, i, function_index) where {T}
    d = get_dirichlet_condition(prob, u, t, i, function_index)
    u[i] = d
    return nothing
end

# get the dirichlet value and update u for a system. need this function barriers for inference
@inline function update_dirichlet_nodes_single!(u::T, t, prob::FVMSystem, i, var, function_index) where {T}
    d = get_dirichlet_condition(prob, u, t, i, var, function_index)
    u[var, i] = d
    return nothing
end

# get the dirichlet value and update u for a non-system for each dirichlet_node
function serial_update_dirichlet_nodes!(u, t, prob::AbstractFVMProblem)
    for (i, function_index) in get_dirichlet_nodes(prob)
        update_dirichlet_nodes_single!(u, t, prob, i, function_index)
    end
    return nothing
end

# get the dirichlet value and update u for a system for each dirichlet_node
function serial_update_dirichlet_nodes!(u, t, prob::FVMSystem) 
    for var in 1:_neqs(prob)
        for (i, function_index) in get_dirichlet_nodes(prob, var)
            update_dirichlet_nodes_single!(u, t, prob, i, var, function_index)
        end
    end
end

# get the dirichlet value and update u for a non-system for each dirichlet_node in parallel
function parallel_update_dirichlet_nodes!(u, t, p, prob::AbstractFVMProblem)
    dirichlet_nodes = p.dirichlet_nodes
    Threads.@threads for i in dirichlet_nodes
        function_index = get_dirichlet_fidx(prob, i)
        update_dirichlet_nodes_single!(u, t, prob, i, function_index)
    end
    return nothing
end

# get the dirichlet value and update u for a system for each dirichlet_node in parallel
function parallel_update_dirichlet_nodes!(u, t, p, prob::FVMSystem) 
    dirichlet_nodes = p.dirichlet_nodes
    for var in 1:_neqs(prob)
        Threads.@threads for i in dirichlet_nodes[var]
            function_index = get_dirichlet_fidx(prob, i, var)
            update_dirichlet_nodes_single!(u, t, prob, i, var, function_index)
        end
    end
    return nothing
end

# the affect! function for updating dirichlet nodes
function update_dirichlet_nodes!(integrator)
    prob, parallel = integrator.p.prob, integrator.p.parallel
    if parallel == Val(false)
        return serial_update_dirichlet_nodes!(integrator.u, integrator.t, prob)
    else
        return parallel_update_dirichlet_nodes!(integrator.u, integrator.t, integrator.p, prob)
    end
    return nothing
end