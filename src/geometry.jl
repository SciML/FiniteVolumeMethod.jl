# Properties of a control volume's intersection with a triangle
struct TriangleProperties
    shape_function_coefficients::NTuple{9,Float64}
    cv_edge_midpoints::NTuple{3,NTuple{2,Float64}}
    cv_normals::NTuple{3,NTuple{2,Float64}}
    cv_edge_lengths::NTuple{3,Float64}
end

"""
    FVMGeometry(tri::Triangulation)

This is a constructor for the [`FVMGeometry`](@ref) struct, which holds the mesh and associated data for the PDE.

It is assumed that all vertices in `tri` are in the triangulation, meaning `v` is in `tri` for each `v` in `each_point_index(tri)`.
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

function FVMGeometry(tri::Triangulation)
    has_ghost = DelaunayTriangulation.has_ghost_triangles(tri)
    has_ghost || add_ghost_triangles!(tri)
    stats = statistics(tri)
    nn = DelaunayTriangulation.num_solid_vertices(stats)
    nt = DelaunayTriangulation.num_solid_triangles(stats)
    cv_volumes = zeros(nn)
    triangle_props = Dict{NTuple{3,Int},TriangleProperties}()
    sizehint!(cv_volumes, nn)
    sizehint!(triangle_props, nt)
    for T in each_solid_triangle(tri)
        i, j, k = indices(T)
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
        n₁x, n₁y = e₁y, -e₁x
        n₂x, n₂y = e₂y, -e₂x
        n₃x, n₃y = e₃y, -e₃x
        ## Now construct the TriangleProperties
        triangle_props[indices(T)] = TriangleProperties(shape_function_coefficients, ((m₁cx, m₁cy), (m₂cx, m₂cy), (m₃cx, m₃cy)), ((n₁x, n₁y), (n₂x, n₂y), (n₃x, n₃y)), (ℓ₁, ℓ₂, ℓ₃))
    end
    has_ghost || delete_ghost_triangles!(tri)
    return FVMGeometry(tri, stats, cv_volumes, triangle_props)
end