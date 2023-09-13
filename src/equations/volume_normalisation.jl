function normalise_du!(du, prob::FVMProblem)
    for i in each_solid_vertex(prob.mesh.triangulation)
        Vᵢ = get_volume(prob, i)
        du[i] /= Vᵢ
    end
    return du
end
function normalise_du!(du, prob::FVMSystem)
    for i in each_solid_vertex(prob.mesh.triangulation)
        Vᵢ = get_volume(prob, i)
        for var in 1:_neqs(prob)
            du[var, i] /= Vᵢ
        end
    end
    return du
end

function parallel_normalise_du!(du, prob::FVMProblem, solid_vertices)
    Base.Threads.@threads for i in solid_vertices
        Vᵢ = get_volume(prob, i)
        du[i] /= Vᵢ
    end
end
function parallel_normalise_du!(du, prob::FVMSystem, solid_vertices)
    Base.Threads.@threads for i in solid_vertices
        Vᵢ = get_volume(prob, i)
        for var in 1:_neqs(prob)
            du[var, i] /= Vᵢ
        end
    end
end