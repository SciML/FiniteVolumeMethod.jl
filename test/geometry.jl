using ..FiniteVolumeMethod
using Test
using LinearAlgebra
using DelaunayTriangulation
function get_control_volume(tri, i)
    is_bnd, bnd_idx = DelaunayTriangulation.is_boundary_node(tri, i)
    cv = NTuple{2,Float64}[]
    if is_bnd
        j = DelaunayTriangulation.get_right_boundary_node(tri, i, bnd_idx)
        k = get_adjacent(tri, i, j)
        p = get_point(tri, i)
        push!(cv, p)
        while !DelaunayTriangulation.is_boundary_index(k)
            q, r = get_point(tri, j, k)
            c = (p .+ q .+ r) ./ 3
            m = (p .+ q) ./ 2
            push!(cv, m, c)
            j = k
            k = get_adjacent(tri, i, j)
            DelaunayTriangulation.is_boundary_index(k) && push!(cv, (p .+ r) ./ 2)
        end
        push!(cv, p)
    else
        S = DelaunayTriangulation.get_surrounding_polygon(tri, i)
        push!(S, S[begin])
        j = S[begin]
        p = get_point(tri, i)
        q = get_point(tri, j)
        push!(cv, (p .+ q) ./ 2)
        for k in S[2:end]
            r = get_point(tri, k)
            push!(cv, (p .+ q .+ r) ./ 3)
            push!(cv, (p .+ r) ./ 2)
            q = r
        end
    end
    return cv
end

a, b, c, d, nx, ny = 0.0, 2.0, 0.0, 5.0, 5, 6
tri = triangulate_rectangle(a, b, c, d, nx, ny; single_boundary=false, add_ghost_triangles=true)
geo = FVMGeometry(tri)
@test geo.triangulation === tri
for i in each_solid_vertex(tri)
    cv = get_control_volume(tri, i)
    A = DelaunayTriangulation.polygon_features(cv,eachindex(cv))[1]
    @test geo.cv_volumes[i] ≈ A
end