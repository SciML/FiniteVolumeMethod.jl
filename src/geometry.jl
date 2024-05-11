# Properties of a control volume's intersection with a triangle
"""
    TriangleProperties(shape_function_coefficients, cv_edge_midpoints, cv_edge_normals, cv_edge_lengths)

This is a struct for holding the properties of a control volume's intersection with a triangle.

# Fields
- `shape_function_coefficients::NTuple{9,Float64}`: The shape function coefficients for the triangle.
- `cv_edge_midpoints::NTuple{3,NTuple{2,Float64}}`: The midpoints of the control volume edges. If the triangle is `(i, j, k)`, then the edges are given in the order `(i, j)`, `(j, k)`, and `(k, i)`, where 'edge' refers to the edge joining e.g. the midpoint of the edge `(i, j)` to the centroid of the triangle. 
- `cv_edge_normals::NTuple{3,NTuple{2,Float64}}`: The normal vectors to the control volume edges, in the same order as in `cv_edge_midpoints`.
- `cv_edge_lengths::NTuple{3,Float64}`: The lengths of the control volume edges, in the same order as in `cv_edge_midpoints`.

!!! notes 

    The shape function coefficients are defined so that, if `s` are the coefficients and the triangle is `T = (i, j, k)`,
    with function values `u[i]`, `u[j]`, and `u[k]` at the vertices `i`, `j`, and `k`, respectively, 
    then `αxₙ + βyₙ + γₙ = u[n]` for `n = i, j, k`, where `xₙ` and `yₙ` are the `x`- and `y`-coordinates of the `n`th vertex, respectively, 
    `α = s₁u₁ + s₂u₂ + s₃u₃`, `β = s₄u₁ + s₅u₂ + s₆u₃`, and `γ = s₇u₁ + s₈u₂ + s₉u₃`.
"""
struct TriangleProperties
    shape_function_coefficients::NTuple{9,Float64}
    cv_edge_midpoints::NTuple{3,NTuple{2,Float64}}
    cv_edge_normals::NTuple{3,NTuple{2,Float64}}
    cv_edge_lengths::NTuple{3,Float64}
end

"""
    FVMGeometry(tri::Triangulation)

This is a constructor for the [`FVMGeometry`](@ref) struct, which holds the mesh and associated data for the PDE.

!!! note
    It is assumed that all vertices in `tri` are in the triangulation, meaning `v` is in `tri` for each `v` in `DelaunayTriangulation.each_point_index(tri)`.

# Fields 
- `triangulation`: The underlying `Triangulation` from DelaunayTriangulation.jl.
- `triangulation_statistics`: The statistics of the triangulation. 
- `cv_volumes::Vector{Float64}`: A `Vector` of the volumes of each control volume.
- `triangle_props::Dict{NTuple{3,Int},TriangleProperties}`: A `Dict` mapping the indices of each triangle to its [`TriangleProperties`].
"""
struct FVMGeometry{T,S}
    triangulation::T
    triangulation_statistics::S
    cv_volumes::Vector{Float64}
    triangle_props::Dict{NTuple{3,Int},TriangleProperties}
end
function Base.show(io::IO, ::MIME"text/plain", geo::FVMGeometry)
    nv = DelaunayTriangulation.num_solid_vertices(geo.triangulation_statistics)
    nt = DelaunayTriangulation.num_solid_triangles(geo.triangulation_statistics)
    ne = DelaunayTriangulation.num_solid_edges(geo.triangulation_statistics)
    print(io, "FVMGeometry with $(nv) control volumes, $(nt) triangles, and $(ne) edges")
end
"""
    get_triangle_props(mesh, i, j, k)

Get the [`TriangleProperties`](@ref) for the triangle `(i, j, k)` in `mesh`.
"""
get_triangle_props(mesh::FVMGeometry, i, j, k) = mesh.triangle_props[(i, j, k)]

"""
    get_volume(mesh, i)

Get the volume of the `i`th control volume in `mesh`.
"""
get_volume(mesh::FVMGeometry, i) = mesh.cv_volumes[i]

"""
    get_point(mesh, i)

Get the `i`th point in `mesh`.
"""
DelaunayTriangulation.get_point(mesh::FVMGeometry, i) = DelaunayTriangulation.get_point(mesh.triangulation, i)

#=
function build_vertex_map(tri::Triangulation)
    vertex_map = Dict{Int,Int}()
    inverse_vertex_map = Dict{Int,Int}()
    cur_idx = 1
    for i in DelaunayTriangulation.each_point_index(tri)
        if DelaunayTriangulation.has_vertex(tri, i)
            vertex_map[i] = cur_idx
            inverse_vertex_map[cur_idx] = i
            cur_idx += 1
        end
    end
    return vertex_map, inverse_vertex_map
end
=#

function FVMGeometry(tri::Triangulation)
    stats = statistics(tri)
    nn = DelaunayTriangulation.num_points(tri)
    nt = num_solid_triangles(stats)
    cv_volumes = zeros(nn)
    triangle_props = Dict{NTuple{3,Int},TriangleProperties}()
    sizehint!(cv_volumes, nn)
    sizehint!(triangle_props, nt)
    for T in each_solid_triangle(tri)
        i, j, k = triangle_vertices(T)
        p, q, r = get_point(tri, i, j, k)
        px, py = getxy(p)
        qx, qy = getxy(q)
        rx, ry = getxy(r)
        ## Get the centroid of the triangle, and the midpoint of each edge
        centroid = DelaunayTriangulation.get_centroid(stats, T)
        m₁, m₂, m₃ = DelaunayTriangulation.get_edge_midpoints(stats, T)
        ## Need to get the sub-control volume areas
        # We need to connect the centroid to each vertex 
        cx, cy = getxy(centroid)
        pcx, pcy = cx - px, cy - py
        qcx, qcy = cx - qx, cy - qy
        rcx, rcy = cx - rx, cy - ry
        # Next, connect all the midpoints to each other
        m₁x, m₁y = getxy(m₁)
        m₂x, m₂y = getxy(m₂)
        m₃x, m₃y = getxy(m₃)
        m₁₃x, m₁₃y = m₁x - m₃x, m₁y - m₃y
        m₂₁x, m₂₁y = m₂x - m₁x, m₂y - m₁y
        m₃₂x, m₃₂y = m₃x - m₂x, m₃y - m₂y
        # We can now contribute the portion of each vertex's control volume inside the triangle to its total volume 
        S₁ = 1 / 2 * abs(pcx * m₁₃y - pcy * m₁₃x)
        S₂ = 1 / 2 * abs(qcx * m₂₁y - qcy * m₂₁x)
        S₃ = 1 / 2 * abs(rcx * m₃₂y - rcy * m₃₂x)
        cv_volumes[i] += S₁
        cv_volumes[j] += S₂
        cv_volumes[k] += S₃
        ## Next, we need to compute the shape function coefficients
        Δ = qx * ry - qy * rx - px * ry + rx * py + px * qy - qx * py
        s₁ = (qy - ry) / Δ
        s₂ = (ry - py) / Δ
        s₃ = (py - qy) / Δ
        s₄ = (rx - qx) / Δ
        s₅ = (px - rx) / Δ
        s₆ = (qx - px) / Δ
        s₇ = (qx * ry - rx * qy) / Δ
        s₈ = (rx * py - px * ry) / Δ
        s₉ = (px * qy - qx * py) / Δ
        shape_function_coefficients = (s₁, s₂, s₃, s₄, s₅, s₆, s₇, s₈, s₉)
        ## Now we need the control volume edge midpoints 
        m₁cx, m₁cy = (m₁x + cx) / 2, (m₁y + cy) / 2
        m₂cx, m₂cy = (m₂x + cx) / 2, (m₂y + cy) / 2
        m₃cx, m₃cy = (m₃x + cx) / 2, (m₃y + cy) / 2
        ## Next, we need the normal vectors to the control volume edges 
        e₁x, e₁y = cx - m₁x, cy - m₁y
        e₂x, e₂y = cx - m₂x, cy - m₂y
        e₃x, e₃y = cx - m₃x, cy - m₃y
        ℓ₁ = norm((e₁x, e₁y))
        ℓ₂ = norm((e₂x, e₂y))
        ℓ₃ = norm((e₃x, e₃y))
        n₁x, n₁y = e₁y / ℓ₁, -e₁x / ℓ₁
        n₂x, n₂y = e₂y / ℓ₂, -e₂x / ℓ₂
        n₃x, n₃y = e₃y / ℓ₃, -e₃x / ℓ₃
        ## Now construct the TriangleProperties
        triangle_props[triangle_vertices(T)] = TriangleProperties(shape_function_coefficients, ((m₁cx, m₁cy), (m₂cx, m₂cy), (m₃cx, m₃cy)), ((n₁x, n₁y), (n₂x, n₂y), (n₃x, n₃y)), (ℓ₁, ℓ₂, ℓ₃))
    end
    return FVMGeometry(tri, stats, cv_volumes, triangle_props)
end