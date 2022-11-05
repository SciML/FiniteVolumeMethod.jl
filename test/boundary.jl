T, adj, adj2v, DG, pts, BN = TestTri()
mesh = FVMGeometry(T, adj, adj2v, DG, pts, BN)

## Classifying boundary edges
_edge_matrix = mesh.boundary_edges
dirichlet_nodes, neumann_nodes, dudt_nodes = FVM.classify_edges(_edge_matrix, [:D, :N, :D, :N])
@test dirichlet_nodes == [1, 5, 2, 3, 8, 4]
@test neumann_nodes == [6, 7, 9, 10]
@test dudt_nodes == Int64[]
dirichlet_nodes, neumann_nodes, dudt_nodes = FVM.classify_edges(_edge_matrix, [:N, :D, :N, :D])
@test dirichlet_nodes == [1, 2, 6, 7, 3, 4, 9, 10]
@test neumann_nodes == [5, 8]
@test dudt_nodes == Int64[]
dirichlet_nodes, neumann_nodes, dudt_nodes = FVM.classify_edges(_edge_matrix, [:N, :N, :N, :N])
@test dirichlet_nodes == Int64[]
@test neumann_nodes == [1, 5, 2, 6, 7, 3, 8, 4, 9, 10]
@test dudt_nodes == Int64[]
dirichlet_nodes, neumann_nodes, dudt_nodes = FVM.classify_edges(_edge_matrix, [:N, :dudt, :dudt, :D])
@test dirichlet_nodes == [1, 4, 9, 10]
@test neumann_nodes == [5]
@test dudt_nodes == [2, 6, 7, 3, 8]

## Boundary index maps 
∂, ∂⁻¹ = FVM.boundary_idx_maps(_edge_matrix)
∂_keys = 1:10
∂_values = [1, 5, 2, 6, 7, 3, 8, 4, 9, 10]
for (i, idx) in enumerate(∂_keys)
    @test ∂[idx] == ∂_values[i]
    @test ∂⁻¹[∂_values[i]] == idx
end
_∂, _∂⁻¹ = FVM.boundary_idx_maps(_edge_matrix, ∂, ∂⁻¹)
@test _∂ == ∂
@test _∂⁻¹ == ∂⁻¹

## Checking function arguments
FNC_TEST_1 = (x, y) -> 0.0
FNC_TEST_2 = (x, y, p) -> 0.0
FNC_TEST_3 = () -> 0.0
FNC_TEST_4 = (a, b, c, d, e) -> 0.0
res = [2, 3, 0, 5]
FNCs = FunctionWrangler([FNC_TEST_1, FNC_TEST_2, FNC_TEST_3, FNC_TEST_4])
for (i, n) in enumerate(res)
    @test FVM.check_nargs(FNCs, i, n)
end

## Type map constructor
type_map = Dict{Int64,Int64}([])
T, adj, adj2v, DG, pts, BN = TestTri()
mesh = FVMGeometry(T, adj, adj2v, DG, pts, BN)
_edge_matrix = mesh.boundary_edges
Dfnc = (x, y, t, p) -> ()
Nfnc = (x, y, p) -> ()
dudtfnc = (x, y, t, u, p) -> ()
dirichlet_nodes, neumann_nodes, dudt_nodes = FVM.classify_edges(_edge_matrix, [:D, :N, :D, :N])
functions = FunctionWrangler([Dfnc, Nfnc, Dfnc, Nfnc])
FVM.construct_type_map!(type_map, dirichlet_nodes, ∂⁻¹, mesh, FVM.DIRICHLET_ARGS, functions)
FVM.construct_type_map!(type_map, neumann_nodes, ∂⁻¹, mesh, FVM.NEUMANN_ARGS, functions)
FVM.construct_type_map!(type_map, dudt_nodes, ∂⁻¹, mesh, FVM.DUDT_ARGS, functions)
ky = [1, 5, 2, 6, 7, 3, 8, 4, 9, 10, 1]
vs = [1, 1, 1, 2, 2, 3, 3, 3, 4, 4]
for (k, v) in zip(ky, vs)
    @test type_map[k] == v
end
dirichlet_nodes, neumann_nodes, dudt_nodes = FVM.classify_edges(_edge_matrix, [:N, :D, :N, :D])
type_map = Dict{Int64,Int64}([])
functions = FunctionWrangler([Nfnc, Dfnc, Nfnc, Dfnc])
FVM.construct_type_map!(type_map, dirichlet_nodes, ∂⁻¹, mesh, FVM.DIRICHLET_ARGS, functions)
FVM.construct_type_map!(type_map, neumann_nodes, ∂⁻¹, mesh, FVM.NEUMANN_ARGS, functions)
FVM.construct_type_map!(type_map, dudt_nodes, ∂⁻¹, mesh, FVM.DUDT_ARGS, functions)
vs = [4, 1, 2, 2, 2, 2, 3, 4, 4, 4]
for (k, v) in zip(ky, vs)
    @test type_map[k] == v
end
dirichlet_nodes, neumann_nodes, dudt_nodes = FVM.classify_edges(_edge_matrix, [:N, :N, :N, :N])
type_map = Dict{Int64,Int64}([])
functions = FunctionWrangler([Nfnc, Nfnc, Nfnc, Nfnc])
FVM.construct_type_map!(type_map, dirichlet_nodes, ∂⁻¹, mesh, FVM.DIRICHLET_ARGS, functions)
FVM.construct_type_map!(type_map, neumann_nodes, ∂⁻¹, mesh, FVM.NEUMANN_ARGS, functions)
FVM.construct_type_map!(type_map, dudt_nodes, ∂⁻¹, mesh, FVM.DUDT_ARGS, functions)
vs = [1, 1, 2, 2, 2, 3, 3, 4, 4, 4]
for (k, v) in zip(ky, vs)
    @test type_map[k] == v
end
dirichlet_nodes, neumann_nodes, dudt_nodes = FVM.classify_edges(_edge_matrix, [:N, :dudt, :dudt, :D])
type_map = Dict{Int64,Int64}([])
functions = FunctionWrangler([Nfnc, dudtfnc, dudtfnc, Dfnc])
FVM.construct_type_map!(type_map, dirichlet_nodes, ∂⁻¹, mesh, FVM.DIRICHLET_ARGS, functions)
FVM.construct_type_map!(type_map, neumann_nodes, ∂⁻¹, mesh, FVM.NEUMANN_ARGS, functions)
FVM.construct_type_map!(type_map, dudt_nodes, ∂⁻¹, mesh, FVM.DUDT_ARGS, functions)
vs = [4, 1, 2, 2, 2, 3, 3, 4, 4, 4]
for (k, v) in zip(ky, vs)
    @test type_map[k] == v
end

## Converting string-specified boundary conditions into symbols 
type = ["Neumann", "Dirichlet", "Neumann"]
new_type = FVM.symbolise_types(type)
@test new_type == (:N, :D, :N)
type = ["Dirichlet"]
new_type = FVM.symbolise_types(type)
@test new_type == (:D,)

type = ["dudt"]
new_type = FVM.symbolise_types(type)
@test new_type == (:dudt,)
type = [:D, :N, :D, :N]
@test FVM.symbolise_types(type) == Tuple(type)
type = (:D, :N, :N)
@test FVM.symbolise_types(type) == type

## Tuple construction 
Dfnc = (x, y, t, p) -> ()
Nfnc = (x, y, p) -> ()
dudtfnc = (x, y, t, u, p) -> ()
T, adj, adj2v, DG, pts, BN = TestTri()
mesh = FVMGeometry(T, adj, adj2v, DG, pts, BN)
functions = [Nfnc, Dfnc, Nfnc, Dfnc]
types = [:N, :D, :N, :D]
dirichlet_nodes, neumann_nodes, dudt_nodes = FVM.classify_edges(_edge_matrix, types)
type_map = Dict{Int64,Int64}([])
FVM.construct_type_map!(type_map, dirichlet_nodes, ∂⁻¹, mesh, FVM.DIRICHLET_ARGS, functions)
FVM.construct_type_map!(type_map, neumann_nodes, ∂⁻¹, mesh, FVM.NEUMANN_ARGS, functions)
FVM.construct_type_map!(type_map, dudt_nodes, ∂⁻¹, mesh, FVM.DUDT_ARGS, functions)
params = [1, 5, 100, 50]
dirichlet_tuples = FVM.node_loop_vals(type_map, params, mesh.points, dirichlet_nodes)
exact_dirichlet_tuples = [
    (1, 4, pts[1, 1], pts[2, 1], params[4]),
    (2, 2, pts[1, 2], pts[2, 2], params[2]),
    (6, 2, pts[1, 6], pts[2, 6], params[2]),
    (7, 2, pts[1, 7], pts[2, 7], params[2]),
    (3, 2, pts[1, 3], pts[2, 3], params[2]),
    (4, 4, pts[1, 4], pts[2, 4], params[4]),
    (9, 4, pts[1, 9], pts[2, 9], params[4]),
    (10, 4, pts[1, 10], pts[2, 10], params[4])
]
for (tup1, tup2) in zip(dirichlet_tuples, exact_dirichlet_tuples)
    @test tup1 == tup2
end
dudt_tuples = FVM.node_loop_vals(type_map, params, mesh.points, dudt_nodes)
@test dudt_tuples == ()

## BoundaryConditions constructor
Dfnc = (x, y, t, p) -> ()
Nfnc = (x, y, p) -> ()
dudtfnc = (x, y, t, u, p) -> ()
T, adj, adj2v, DG, pts, BN = TestTri()
mesh = FVMGeometry(T, adj, adj2v, DG, pts, BN)
functions = [Nfnc, Dfnc, Nfnc, Dfnc]
types = (:N, :D, :N, :D)
dirichlet_nodes, neumann_nodes, dudt_nodes = FVM.classify_edges(_edge_matrix, types)
type_map = Dict{Int64,Int64}([])
FVM.construct_type_map!(type_map, dirichlet_nodes, ∂⁻¹, mesh, FVM.DIRICHLET_ARGS, functions)
FVM.construct_type_map!(type_map, neumann_nodes, ∂⁻¹, mesh, FVM.NEUMANN_ARGS, functions)
FVM.construct_type_map!(type_map, dudt_nodes, ∂⁻¹, mesh, FVM.DUDT_ARGS, functions)
BCs = BoundaryConditions(mesh, functions, types)
@test BCs.boundary_node_vector == BN
@test BCs.functions == FunctionWrangler(functions)
@test BCs.type == types
@test BCs.params == Tuple(Float64[] for _ in 1:length(functions))
@test BCs.dirichlet_nodes == dirichlet_nodes
@test BCs.neumann_nodes == neumann_nodes
@test BCs.dudt_nodes == dudt_nodes
@test BCs.interior_or_neumann_nodes == union(mesh.interior_points, neumann_nodes)
@test BCs.boundary_to_mesh_idx == ∂
@test BCs.mesh_to_boundary_idx == ∂⁻¹
@test BCs.type_map == type_map
BCs2 = BoundaryConditions(mesh, functions, types;
    params=BCs.params,
    dirichlet_nodes,
    neumann_nodes,
    dudt_nodes,
    interior_or_neumann_nodes=union(mesh.interior_points, neumann_nodes),
    ∂,
    ∂⁻¹,
    type_map)
propnames = propertynames(BCs)
@test all(getproperty(BCs, name) == getproperty(BCs2, name) for name in propnames)
