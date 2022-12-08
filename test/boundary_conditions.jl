## Need to start by testing the wrapper functions 
# Test the argument type constructors 
for T in (Float64, Float32, Int64, Float16)
    for U in (Float64, Float32, Float16, Int64)
        for P in (Nothing, Float64, Vector{Float64}, Float32, NTuple{3,Float64}, Tuple{Nothing,Float64,Float32})
            T = Float64
            U = Float64
            P = Nothing
            dT = dualgen(T)
            dU = dualgen(U)
            arg_types = FVM.get_dual_arg_types(T, U, P)
            @test arg_types == (Tuple{T,T,T,U,P}, Tuple{T,T,T,dU,P}, Tuple{T,T,dT,U,P}, Tuple{T,T,dT,dU,P})
            ret_types = FVM.get_dual_ret_types(U, T)
            @test ret_types == (U, dU, dT, promote_type(dT, dU))
        end
    end
end

# Define the functions
dirichlet_f1 = (x, y, t, u, p) -> x^2 + y^2
dirichlet_f2 = (x, y, t, u, p) -> x * p[1] * p[2] + u
neumann_f1 = (x, y, t, u, p) -> 0.0
neumann_f2 = (x, y, t, u, p) -> u * p[1]
dudt_dirichlet_f1 = (x, y, t, u, p) -> x * y + 1
dudt_dirichlet_f2 = (x, y, t, u, p) -> u + y * t
p = (nothing, (3.3, 4.3), nothing, [0.2], 0.0, 5)
#p = ((3.0,3.0),(3.0,3.2),(3.0,3.2),(3.0,3.2),(3.0,3.2),(3.0,3.2))

# Test the construction of the wrappers
float_type = Float64
u_type = Float64
function_list = (dirichlet_f1, dirichlet_f2, neumann_f1, neumann_f2, dudt_dirichlet_f1, dudt_dirichlet_f2)
types = (:D, :D, :N, :N, :dudt, :dudt)
wrappers = FVM.wrap_functions(function_list, p; float_type, u_type)
for (wrapper, fnc, _p) in zip(wrappers, function_list, p)
    x, y, t, u = rand(4)
    @test wrapper(x, y, t, u, _p) ≈ fnc(x, y, t, u, _p)
end

# Test the type-stability of the wrappers 
for (wrapper, fnc, _p) in zip(wrappers, function_list, p)
    x, y, t, u = rand(4)
    @inferred wrapper(x, y, t, u, _p) ≈ fnc(x, y, t, u, _p)
end

# Test the type-stability for a vector of functions 
function vector_of_function_test(wrappers, parameters)
    nums = zeros(length(wrappers))
    @inline map(wrappers, eachindex(wrappers), parameters) do f, i, p
        x, y, t, u = rand(4)
        nums[i] = f(x, y, t, u, p)
    end
    return nums
end
wrapper_vec = (wrappers[1], wrappers[2])
@inferred vector_of_function_test(wrapper_vec, p)
SHOW_WARNTYPE && @code_warntype vector_of_function_test(wrapper_vec, p)

wrapper_vec = (wrappers[3], wrappers[4])
@inferred vector_of_function_test(wrapper_vec, p[3:4])
SHOW_WARNTYPE && @code_warntype vector_of_function_test(wrapper_vec, p[3:4])

wrapper_vec = (wrappers[5], wrappers[6])
@inferred vector_of_function_test(wrapper_vec, p[5:6])
SHOW_WARNTYPE && @code_warntype vector_of_function_test(wrapper_vec, p[5:6])

wrapper_vec = wrappers
@inferred vector_of_function_test(wrapper_vec, p)
SHOW_WARNTYPE && @code_warntype vector_of_function_test(wrapper_vec, p)

## Make sure we can test the types correctly 
# For a single provided type 
for type in [:Dirichlet, :D, :dirichlet, "Dirichlet", "D", "dirichlet"]
    @test FVM.is_dirichlet_type(type)
end
for type in [:Neumann, :N, :neumann, "Neumann", "N", "neumann"]
    @test FVM.is_neumann_type(type)
end
for type in [:Dudt, :dudt, "Dudt", "dudt", "du/dt"]
    @test FVM.is_dudt_type(type)
end

# For merging two types 
@test FVM.classify_edge_type(:D, :D) == :D
@test FVM.classify_edge_type(:D, :N) == :D
@test FVM.classify_edge_type(:D, :dudt) == :D
@test FVM.classify_edge_type(:N, :D) == :D
@test FVM.classify_edge_type(:dudt, :D) == :D
@test FVM.classify_edge_type(:dudt, :N) == :dudt
@test FVM.classify_edge_type(:dudt, :dudt) == :dudt
@test FVM.classify_edge_type(:N, :dudt) == :dudt
@test FVM.classify_edge_type(:N, :N) == :N

## Check that we can correctly classify the edges 
a, b, c, d, nx, ny, T, adj, adj2v, DG, pts, BN = example_triangulation()
geo = FVMGeometry(T, adj, adj2v, DG, pts, BN)
edge_information = FVM.get_boundary_edge_information(geo)
edge_types = [:N, :D, :dudt, :D]
dirichlet_nodes, neumann_nodes, dudt_nodes = FVM.classify_edges(edge_information, edge_types)
@test dirichlet_nodes == [1, 5, 10, 15, 20, 25, 30, 26, 21, 16, 11, 6]
@test neumann_nodes == [2, 3, 4]
@test dudt_nodes == [29, 28, 27]

edge_types = [:N, :N, :N, :N]
dirichlet_nodes, neumann_nodes, dudt_nodes = FVM.classify_edges(edge_information, edge_types)
@test dirichlet_nodes == Int64[]
@test neumann_nodes == [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 29, 28, 27, 26, 21, 16, 11, 6]
@test dudt_nodes == Int64[]

edge_types = [:dudt, :dudt, :dudt, :dudt]
dirichlet_nodes, neumann_nodes, dudt_nodes = FVM.classify_edges(edge_information, edge_types)
@test dirichlet_nodes == Int64[]
@test neumann_nodes == Int64[]
@test dudt_nodes == [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 29, 28, 27, 26, 21, 16, 11, 6]

## Check that we can correctly map between boundary bases 
∂, ∂⁻¹ = FVM.boundary_index_maps(edge_information)
true_boundary_nodes = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 29, 28, 27, 26, 21, 16, 11, 6]
true_boundary_order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
for i in eachindex(edge_information)
    @test ∂[true_boundary_order[i]] == true_boundary_nodes[i]
    @test ∂⁻¹[true_boundary_nodes[i]] == true_boundary_order[i]
end

## Test that we can correctly assign each node a segment 
edge_types = [:N, :D, :dudt, :D]
boundary_nodes = FVM.get_boundary_nodes(edge_information)
f1 = neumann_f1
f2 = dirichlet_f2
f3 = dudt_dirichlet_f2
f4 = dirichlet_f1
functions = (f1, f2, f3, f4)
params = (nothing, 1.0, nothing, 1.0)
dirichlet_nodes, neumann_nodes, dudt_nodes = FVM.classify_edges(edge_information, edge_types)
type_map = FVM.construct_type_map(dirichlet_nodes, neumann_nodes, dudt_nodes, ∂⁻¹, edge_information, edge_types)
true_segment_pairs = [
    (2, 1), (3, 1), (4, 1),
    (5, 2), (10, 2), (15, 2), (20, 2), (25, 2), (30, 2),
    (29, 3), (28, 3), (27, 3),
    (26, 4), (21, 4), (16, 4), (11, 4), (6, 4), (1, 4)
]
for (k, v) in true_segment_pairs
    @test type_map[k] == v
end

## Check that we get the boundary condition object correct 
functions = (dirichlet_f2, neumann_f1, dudt_dirichlet_f1, dudt_dirichlet_f2)
types = (:D, :N, :dudt, :dudt)
boundary_node_vector = BN
params = ((1.0, 2.0), nothing, nothing, nothing)
BCs = FVM.BoundaryConditions(geo, functions, types, boundary_node_vector; params)
@test BCs.boundary_node_vector == BN
x, y, t, u = rand(4)
@test all(BCs.functions[i](x, y, t, u, params[i]) == functions[i](x, y, t, u, params[i]) for i in eachindex(functions))
@test BCs.dirichlet_nodes == [1, 2, 3, 4, 5]
@test BCs.neumann_nodes == [10, 15, 20, 25]
@test BCs.dudt_nodes == [30, 29, 28, 27, 26, 21, 16, 11, 6]
@test BCs.interior_or_neumann_nodes == Set{Int64}([7, 8, 9, 12, 13, 14, 17, 18, 19, 22, 23, 24, 10, 15, 20, 25])
true_boundary_nodes = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 29, 28, 27, 26, 21, 16, 11, 6]
true_boundary_order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
for i in eachindex(edge_information)
    @test BCs.boundary_to_mesh_idx[true_boundary_order[i]] == true_boundary_nodes[i]
    @test BCs.mesh_to_boundary_idx[true_boundary_nodes[i]] == true_boundary_order[i]
end
true_segment_pairs = [
    (1, 1), (2, 1), (3, 1), (4, 1), (5, 1),
    (10, 2), (15, 2), (20, 2), (25, 2),
    (30, 3), (29, 3), (28, 3), (27, 3),
    (26, 4), (21, 4), (16, 4), (11, 4), (6, 4)
]
for (k, v) in true_segment_pairs
    @test BCs.type_map[k] == v
end