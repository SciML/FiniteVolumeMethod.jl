# the main entry point
"""
    fvm_eqs!(du, u, p, t)

Computes the finite volume method equations for the current time `t` 
and solution `u`. This function is public API.

The argument `p` depends on whether the problem is being solved 
in parallel or not. If it is solved serially, than the fields are:

- `p.prob`: The `prob <: AbstractFVMProblem`.
- `p.parallel`: `Val(false)`.

If the problem is solved in parallel, then the fields are:

- `p.prob`: The `prob <: AbstractFVMProblem`.
- `p.parallel`: `Val(true)`.   
- `p.duplicated_du`: A `Matrix` of the same size as `du` that is used to store the contributions to `du` from each thread. 
- `p.solid_triangles`: A `Vector` of the solid triangles in the triangulation.
- `p.solid_vertices`: A `Vector` of the solid vertices in the triangulation.
- `p.chunked_solid_triangles`: A `Vector` of tuples of the form `(range, chunk_idx)` where `range` is a range of indices into `p.solid_triangles` and `chunk_idx` is the index of the chunk.
- `p.boundary_edges`: A `Vector` of the boundary edges in the triangulation.
- `p.chunked_boundary_edges`: A `Vector` of tuples of the form `(range, chunk_idx)` where `range` is a range of indices into `p.boundary_edges` and `chunk_idx` is the index of the chunk.

These fields are public API, although note that they are not intended to be modified by the user, and we may freely 
add in new fields over new versions.
"""
function fvm_eqs!(du, u, p, t)
    prob, parallel = p.prob, p.parallel
    if parallel == Val(false)
        return serial_fvm_eqs!(du, u, prob, t)
    else
        return parallel_fvm_eqs!(du, u, p, t)
    end
end

# the serial form of the equations
function serial_fvm_eqs!(du, u, prob, t)
    fill!(du, zero(eltype(du)))
    get_triangle_contributions!(du, u, prob, t) # acts on triangles 
    get_boundary_edge_contributions!(du, u, prob, t) # acts on boundary edges
    get_source_contributions!(du, u, prob, t) # acts on points
    return du
end

# the parallel form of the equations
function parallel_fvm_eqs!(du, u, p, t)
    duplicated_du, solid_triangles,
    solid_vertices, chunked_solid_triangles,
    boundary_edges, chunked_boundary_edges,
    prob = p.duplicated_du, p.solid_triangles,
    p.solid_vertices, p.chunked_solid_triangles,
    p.boundary_edges, p.chunked_boundary_edges,
    p.prob
    fill!(du, zero(eltype(du)))
    _duplicated_du = get_tmp(duplicated_du, du)
    fill!(_duplicated_du, zero(eltype(du)))
    get_parallel_triangle_contributions!(_duplicated_du, u, prob, t, chunked_solid_triangles, solid_triangles)
    get_parallel_boundary_edge_contributions!(_duplicated_du, u, prob, t, chunked_boundary_edges, boundary_edges)
    combine_duplicated_du!(du, _duplicated_du, prob)
    get_parallel_source_contributions!(du, u, prob, t, solid_vertices)
    return du
end

# after the parallel stages are done, we need to combine du back into its original form
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