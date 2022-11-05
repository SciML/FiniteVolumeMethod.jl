T, adj, adj2v, DG, pts, BN = TestTri()

## Centroid 
centroids = Dict{NTuple{3,Int64},Vector{Float64}}()
pt = ElasticMatrix{Float64}([
    1.0 7.0
    -0.5 0.5
])
pt = (_get_point(pt, 1), _get_point(pt, 2))
FVM.centroid!(centroids, (7, 3, 12), pt)
@test centroids[(7, 3, 12)] ≈ [4.0, 0.0]

## Midpoint 
midpoints = Dict{NTuple{3,Int64},NTuple{3,Vector{Float64}}}()
V = (7, 3, 12)
pt = (_get_point(pts, 7), _get_point(pts, 3), _get_point(pts, 12))
FVM.midpoint!(midpoints, V, pt)
true_m = (
    [5.0, 8.333333236866126],
    [3.819444473548958, 8.749999997848507],
    [3.819444473548958, 7.083333234714633],
)
@test collect(midpoints[V]) ≈ collect(true_m)

## Edge vectors 
_edges = Vector{Vector{Float64}}(undef, 3)
FVM.control_volume_edges!(_edges, centroids, midpoints, V)
true_edges = [
    [-1.0, -8.333333236866126],
    [0.1805555264510419, -8.749999997848507],
    [0.1805555264510419, -7.083333234714633]
]
@test _edges ≈ true_edges

## Edge lengths 
L = Dict{NTuple{3,Float64},NTuple{3,Float64}}()
FVM.edge_lengths!(L, V, _edges)
@test collect(L[V]) ≈ [norm(e) for e in _edges]
@test collect(L[V]) ≈ [
    8.393118778896058,
    8.751862673767276,
    7.08563405858254
]

## Edge normals
_normals = Dict{NTuple{3,Float64},NTuple{3,Vector{Float64}}}([])
_edges = true_edges
L = Dict{NTuple{3,Int64},NTuple{3,Float64}}([])
L[V] = Tuple(norm(e) for e in _edges)
FVM.edge_normals!(_normals, V, _edges, L)
true_normals = [[-8.333333236866126, 1.0] / L[V][1],
    [-8.749999997848507, -0.1805555264510419] / L[V][2],
    [-7.083333234714633, -0.1805555264510419] / L[V][3]]
@test collect(true_normals) ≈ collect(_normals[V])

## Control volume node centroids 
p = [zeros(2) for _ in 1:3]
pt = (_get_point(pts, 7), _get_point(pts, 3), _get_point(pts, 12))
FVM.control_volume_node_centroid!(p, centroids, V, pt)
true_p = [[-1.0, -6.666666473732254],
    [-1.0, -10.0],
    [1.3611110529020838, -7.499999995697014]]
@test p ≈ true_p

## Control volume connect midpoints 
q = [zeros(2) for _ in 1:3]
FVM.control_volume_connect_midpoints!(q, midpoints, V)
true_q = [[1.180555526451042, 1.2500000021514932],
    [-1.180555526451042, 0.4166667609823804],
    [0.0, -1.6666667631338736]]
@test q ≈ true_q

## Sub control volume areas
S = zeros(3)
FVM.sub_control_volume_areas!(S, true_p, true_q)
true_S = [
    3.3101849732094992,
    6.1111110127464,
    1.1342592764030273
]
@test S ≈ true_S

## Shape function coefficients 
# Construction for a single shape function
s = Dict{NTuple{3,Int64},NTuple{9,Float64}}([])
pts = [[1.0, 0.5], [5.1, 0.1], [9.0, 9.371]]
FVM.shape_function_coefficients!(s, (1, 2, 3), pts)
u = [1.0, 0.1, 50.0]
α = s[(1, 2, 3)][1:3] |> collect
β = s[(1, 2, 3)][4:6] |> collect
γ = s[(1, 2, 3)][7:9] |> collect
shape_fnc = (xy) -> α' * u * xy[1] + β' * u * xy[2] + γ' * u
@test collect(shape_fnc.(pts)) ≈ u
@test collect(s[(1, 2, 3)]) ≈ [-0.234287, 0.224179, 0.0101084, 0.0985568, -0.202168, 0.103611, 1.18501, -0.123095, -0.0619139] atol = 1e-5
@test sum(s[(1, 2, 3)]) ≈ 1.0

# Testing many shape functions
r = θ -> sin(3θ) + cos(5θ) - sin(θ)
θ = LinRange(0, 2π, 1000)
x = @. (1 + 0.05 * r(θ)) * cos(θ)
y = @. (1 + 0.05 * r(θ)) * sin(θ)
T, adj, adj2v, DG, pts, BN = generate_mesh(x, y, 0.25; gmsh_path=GMSH_PATH)
no_elements = num_triangles(T)
s = Dict{NTuple{3,Int64},NTuple{9,Float64}}([])
Random.seed!(299911)
for τ in T
    local pt
    i, j, k = indices(τ)
    pt = (_get_point(pts, i), _get_point(pts, j), _get_point(pts, k))
    local u = rand(3)
    FVM.shape_function_coefficients!(s, τ, pt)
    local α = s[τ][1:3] |> collect
    local β = s[τ][4:6]|> collect
    local γ = s[τ][7:9]|> collect
    local shape_fnc = (xy) -> α' * u * xy[1] + β' * u * xy[2] + γ' * u
    @test [shape_fnc.(pt)...] ≈ u
    @test sum(s[τ]) ≈ 1.0
end

## Edge matrix
T, adj, adj2v, DG, pts, BN = TestTri()
E = FVM.boundary_edge_matrix(adj, BN)
E1 = [1, 5, 2, 6, 7, 3, 8, 4, 9, 10]
E2 = [5, 2, 6, 7, 3, 8, 4, 9, 10, 1]
E3 = [13, 13, 13, 11, 12, 12, 12, 12, 11, 13]
E5 = [1, 1, 2, 2, 2, 3, 3, 4, 4, 4]
E_true = [[E1[i], E2[i], E3[i], E5[i]] for i in eachindex(E1)]
@test E == E_true

## Outward normal boundary 
ONB = FVM.outward_normal_boundary(pts, E)
b, r, t, l = [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]
@test typeof(ONB) == Vector{Tuple{Vector{Float64},Int64}}
@test [ONB[i][1] for i in eachindex(ONB)] ≈ [b, b, r, r, r, t, t, l, l, l]
@test [ONB[i][2] for i in eachindex(ONB)] == [1, 1, 2, 2, 2, 3, 3, 4, 4, 4]

## Boundary information 
_boundary_edges, _boundary_normals, _boundary_nodes, _boundary_elements = FVM.boundary_information(T, adj, pts, BN)
@test _boundary_edges == E
@test _boundary_normals == ONB
@test _boundary_nodes == [E[i][1] for i in eachindex(E)]
boundary_elements = Set{NTuple{3,Int64}}([
    (1, 5, 13),
    (5, 2, 13),
    (2, 6, 13),
    (6, 7, 11),
    (4, 9, 12),
    (9, 10, 11),
    (10, 1, 13),
    (7, 3, 12),
    (3, 8, 12),
    (8, 4, 12)
])
@test DelaunayTriangulation.compare_triangle_sets(_boundary_elements, boundary_elements)

## Complete construction
T, adj, adj2v, DG, pts, BN = TestTri()
fvm_mesh = FVMGeometry(T, adj, adj2v, DG, pts, BN)
@test fvm_mesh.elements == T
@test fvm_mesh.adj == adj
@test fvm_mesh.adj2v == adj2v
@test fvm_mesh.neighbours == DG
@test fvm_mesh.points == pts
@test fvm_mesh.boundary_node_vector == BN
@test sum(fvm_mesh.volumes |> values) ≈ 5 * 10
@test collect.(fvm_mesh.shape_function_coeffs |> values |> collect) |> sum |> sum ≈ num_triangles(T)
@test sum(fvm_mesh.areas |> values) ≈ 5 * 10
@test fvm_mesh.interior_points == [11, 12, 13]
@test DelaunayTriangulation.compare_triangle_sets(fvm_mesh.interior_elements,
    Set{NTuple{3,Int64}}([
        (11, 10, 13),
        (9, 11, 12),
        (11, 7, 12),
        (6, 11, 13)
    ]))
@test DelaunayTriangulation.compare_triangle_sets(fvm_mesh.boundary_elements,
    Set{NTuple{3,Int64}}([
        (1, 5, 13),
        (5, 2, 13),
        (2, 6, 13),
        (6, 7, 11),
        (7, 3, 12),
        (3, 8, 12),
        (8, 4, 12),
        (4, 9, 12),
        (9, 10, 11),
        (10, 1, 13)
    ]))
@test fvm_mesh.total_area ≈ 5 * 10
for τ in T
    i, j, k = indices(τ)
    p1, p2, p3 = _get_point(pts, i), _get_point(pts, j), _get_point(pts, k)
    @test fvm_mesh.centroids[τ] ≈ (p1 + p2 + p3) / 3
    @test fvm_mesh.weighted_areas[τ] ≈ fvm_mesh.areas[τ] / fvm_mesh.total_area
end
@test sum(fvm_mesh.weighted_areas |> values) ≈ 1.0
