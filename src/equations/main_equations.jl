function fvm_eqs_flux!(du, u, p, t)
    parallel = p.parallel
    if parallel == Val(false)
        return serial_fvm_eqs_flux!(du, u, p, t)
    else
        return parallel_fvm_eqs_flux!(du, u, p, t)
    end
end

function serial_fvm_eqs_flux!(du, u, p, t)
    prob = p.prob
    _serial_fvm_eqs_flux!(du, u, prob, t)
    return du
end
function parallel_fvm_eqs_flux!(du, u, p, t)
    _parallel_fvm_eqs_flux!(du, u, p, t)
    return du
end

function fvm_eqs_source!(du, u, p, t)
    parallel = p.parallel
    if parallel == Val(false)
        return serial_fvm_eqs_source!(du, u, p, t)
    else
        return parallel_fvm_eqs_source!(du, u, p, t)
    end
end

function serial_fvm_eqs_source!(du, u, p, t)
    prob = p.prob
    _serial_fvm_eqs_source!(du, u, prob, t)
    return du
end
function parallel_fvm_eqs_source!(du, u, p, t)
    _parallel_fvm_eqs_source!(du, u, p, t)
    return du
end

function _serial_fvm_eqs_flux!(du, u, prob, t)
    fill!(du, zero(eltype(du)))
    get_triangle_contributions!(du, u, prob, t) # acts on triangles 
    get_boundary_edge_contributions!(du, u, prob, t) # acts on boundary edges
    normalise_du!(du, prob)
    return du
end
function _parallel_fvm_eqs_flux!(du, u, p, t)
    duplicated_du, solid_triangles,
    chunked_solid_triangles,
    boundary_edges, chunked_boundary_edges,
    prob, solid_vertices = p.duplicated_du, p.solid_triangles,
    p.chunked_solid_triangles,
    p.boundary_edges, p.chunked_boundary_edges,
    p.prob, p.solid_vertices
    fill!(du, zero(eltype(du)))
    _duplicated_du = get_tmp(duplicated_du, du)
    fill!(_duplicated_du, zero(eltype(du)))
    get_parallel_triangle_contributions!(_duplicated_du, u, prob, t, chunked_solid_triangles, solid_triangles)
    get_parallel_boundary_edge_contributions!(_duplicated_du, u, prob, t, chunked_boundary_edges, boundary_edges)
    combine_duplicated_du!(du, _duplicated_du, prob)
    parallel_normalise_du!(du, prob, solid_vertices)
    return du
end
function combine_duplicated_du!(du, duplicated_du, ::FVMSystem)
    for i in axes(duplicated_du, 3)
        du .+= duplicated_du[:, :, i]
    end
    return du
end
function combine_duplicated_du!(du, duplicated_du, ::FVMProblem)
    for _du in eachcol(duplicated_du)
        du .+= _du
    end
    return du
end

function _serial_fvm_eqs_source!(du, u, prob, t)
    fill!(du, zero(eltype(du)))
    get_source_contributions!(du, u, prob, t) # acts on points
    return du
end
function _parallel_fvm_eqs_source!(du, u, p, t)
    solid_vertices, prob = p.solid_vertices, p.prob
    fill!(du, zero(eltype(du)))
    get_parallel_source_contributions!(du, u, prob, t, solid_vertices)
    return du
end


