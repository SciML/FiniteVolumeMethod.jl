using ..FiniteVolumeMethod
using Test
using LinearAlgebra
using DelaunayTriangulation
using StructEquality
@struct_hash_equal DelaunayTriangulation.TriangulationStatistics
@struct_hash_equal DelaunayTriangulation.IndividualTriangleStatistics
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
function random_point_inside_triangle(p, q, r)
    b = q .- p
    c = r .- p
    outside = true
    y = (NaN, NaN)
    while outside
        a₁, a₂ = rand(2)
        x = a₁ .* b .+ a₂ .* c
        y = p .+ x
        outside = DelaunayTriangulation.is_outside(
            DelaunayTriangulation.point_position_relative_to_triangle(p, q, r, y)
        )
    end
    return y
end

a, b, c, d, nx, ny = 0.0, 2.0, 0.0, 5.0, 5, 6
tri = triangulate_rectangle(a, b, c, d, nx, ny; single_boundary=false, add_ghost_triangles=true)
geo = FVMGeometry(tri)
@inferred FVMGeometry(tri)
@test geo.triangulation === tri
@test geo.triangulation_statistics == statistics(tri)
for i in each_solid_vertex(tri)
    cv = get_control_volume(tri, i)
    A = DelaunayTriangulation.polygon_features(cv, eachindex(cv))[1]
    @test geo.cv_volumes[i] ≈ A
end
@test length(geo.triangle_props) == DelaunayTriangulation.num_solid_triangles(tri)
for T in each_solid_triangle(tri)
    local a, b, c
    i, j, k = T
    props = geo.triangle_props[(i,j,k)]
    p, q, r = get_point(tri, i, j, k)
    # The shape function coefficients
    u = rand(3) 
    A = [p[1] p[2] 1.0; q[1] q[2] 1.0; r[1] r[2] 1.0]
    α, β, γ = A \ u 
    s = props.shape_function_coefficients
    a = s[1] * u[1] + s[2] * u[2] + s[3] * u[3]
    b = s[4] * u[1] + s[5] * u[2] + s[6] * u[3]
    c = s[7] * u[1] + s[8] * u[2] + s[9] * u[3]
    @test a ≈ α
    @test b ≈ β
    @test c ≈ γ
    @test sum(s) ≈ 1
    x = random_point_inside_triangle(p, q, r)
    f1 = α * x[1] + β * x[2] + γ
    f2 = a * x[1] + b * x[2] + c
    @test f1 ≈ f2
    # The edge midpoints
    centroid = (p .+ q .+ r) ./ 3 
    ij_mid = (p .+ q) ./ 2 
    jk_mid = (q .+ r) ./ 2
    ki_mid = (r .+ p) ./ 2
    cij_mid = (centroid .+ ij_mid) ./ 2 
    cjk_mid = (centroid .+ jk_mid) ./ 2
    cki_mid = (centroid .+ ki_mid) ./ 2
    m1, m2, m3 = props.cv_edge_midpoints 
    @test collect(cij_mid) ≈ collect(m1)
    @test collect(cjk_mid) ≈ collect(m2)
    @test collect(cki_mid) ≈ collect(m3)
    # The edge normals 
    e1 = centroid .- ij_mid
    e2 = centroid .- jk_mid
    e3 = centroid .- ki_mid
    n1 = [e1[2], -e1[1]] / norm(e1)
    n2 = [e2[2], -e2[1]] / norm(e2)
    n3 = [e3[2], -e3[1]] / norm(e3)
    nn1, nn2, nn3 = props.cv_edge_normals
    @test collect(n1) ≈ collect(nn1)
    @test collect(n2) ≈ collect(nn2)
    @test collect(n3) ≈ collect(nn3)
    # The edge lengths 
    ℓ1 = norm(e1)
    ℓ2 = norm(e2)
    ℓ3 = norm(e3)
    L1, L2, L3 = props.cv_edge_lengths
    @test ℓ1 ≈ L1
    @test ℓ2 ≈ L2
    @test ℓ3 ≈ L3
end