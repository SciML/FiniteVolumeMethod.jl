using ..FiniteVolumeMethod
using Test
include("test_setup.jl")
using PreallocationTools
using SparseArrays
using LinearAlgebra

## Define the example problem 
a, b, c, d, nx, ny, tri = example_triangulation()
for coordinate_type in (NTuple{2,Float64}, SVector{2,Float64})
    for control_volume_storage_type_vector in (Vector{coordinate_type}, SVector{3,coordinate_type})
        for control_volume_storage_type_scalar in (Vector{Float64}, SVector{3,Float64})
            for shape_function_coefficient_storage_type in (Vector{Float64}, NTuple{9,Float64})
                for interior_edge_storage_type in (Vector{Int64}, NTuple{2,Int64},)
                    for interior_edge_pair_storage_type in (Vector{interior_edge_storage_type}, NTuple{2,interior_edge_storage_type})
                        geo = FVMGeometry(
                            tri;
                            coordinate_type, control_volume_storage_type_vector,
                            control_volume_storage_type_scalar, shape_function_coefficient_storage_type,
                            interior_edge_storage_type, interior_edge_pair_storage_type)
                        dirichlet_f1 = (x, y, t, u, p) -> x^2 + y^2
                        dirichlet_f2 = (x, y, t, u, p) -> x * p[1] * p[2] + u
                        neumann_f1 = (x, y, t, u, p) -> 0.0
                        neumann_f2 = (x, y, t, u, p) -> u * p[1]
                        dudt_dirichlet_f1 = (x, y, t, u, p) -> x * y + 1
                        dudt_dirichlet_f2 = (x, y, t, u, p) -> u + y * t
                        functions = (dirichlet_f2, neumann_f1, dudt_dirichlet_f1, dudt_dirichlet_f2)
                        types = (:D, :N, :dudt, :dudt)
                        params = ((1.0, 2.0), nothing, nothing, nothing)
                        BCs = FVM.BoundaryConditions(geo, functions, types; params)
                        iip_flux = true
                        flux_function = (q, x, y, t, α, β, γ, p) -> (q[1] = x * y * t; q[2] = t; nothing)
                        initial_condition = zeros(20)
                        final_time = 5.0
                        prob = FVMProblem(geo, BCs; iip_flux, flux_function, initial_condition, final_time)

                        ## Now do some testing 
                        shape_cache = zeros(3)
                        u = rand(num_points(tri))
                        pts = get_points(tri)
                        for V in each_solid_triangle(tri)
                            FVM.linear_shape_function_coefficients!(shape_cache, u, prob, V)
                            interp = (x, y) -> shape_cache[1] * x + shape_cache[2] * y + shape_cache[3]
                            @test interp(pts[1, geti(V)], pts[2, geti(V)]) ≈ u[geti(V)]
                            @test interp(pts[1, getj(V)], pts[2, getj(V)]) ≈ u[getj(V)]
                            @test interp(pts[1, getk(V)], pts[2, getk(V)]) ≈ u[getk(V)]
                        end

                        du = zeros(length(u))
                        t = rand()
                        V = (9, 5, 10)
                        vj, j = 9, 1 # from T = (9, 5, 10)
                        vjnb, jnb = 5, 2
                        α, β, γ = rand(3)
                        flux_cache = zeros(2)
                        FVM.fvm_eqs_edge!(du, t, (vj, j), (vjnb, jnb), α, β, γ, prob, flux_cache, V)
                        @test du[vj] ≈ -(flux_cache[1] * prob.mesh.element_information_list[V].normals[j][1] + flux_cache[2] * prob.mesh.element_information_list[V].normals[j][2]) * prob.mesh.element_information_list[V].lengths[j]
                        @test du[vjnb] == 0.0 # !is_interior_or_neumann_node(prob, vjnb)
                        vj, j, vjnb, jnb = vjnb, jnb, vj, j
                        du = zeros(length(u))
                        FVM.fvm_eqs_edge!(du, t, (vj, j), (vjnb, jnb), α, β, γ, prob, flux_cache, V)
                        @test du[vj] == 0.0
                        @test du[vjnb] ≈ (flux_cache[1] * prob.mesh.element_information_list[V].normals[j][1] + flux_cache[2] * prob.mesh.element_information_list[V].normals[j][2]) * prob.mesh.element_information_list[V].lengths[j]

                        du = zeros(length(u))
                        t = rand()
                        V = (9, 5, 10)
                        α, β, γ = rand(3)
                        flux_cache = zeros(2)
                        FVM.fvm_eqs_edge!(du, t, α, β, γ, prob, flux_cache, V)
                        _du = zeros(length(u))
                        FVM.fvm_eqs_edge!(_du, t, (9, 1), (5, 2), α, β, γ, prob, flux_cache, V)
                        FVM.fvm_eqs_edge!(_du, t, (5, 2), (10, 3), α, β, γ, prob, flux_cache, V)
                        FVM.fvm_eqs_edge!(_du, t, (10, 3), (9, 1), α, β, γ, prob, flux_cache, V)
                        @test du == _du

                        du = zeros(length(u))
                        t = rand()
                        V = (9, 5, 10)
                        shape_coeffs = zeros(3)
                        flux_cache = zeros(2)
                        FVM.fvm_eqs_interior_element!(du, u, t, prob, shape_coeffs, flux_cache, V)
                        _du = zeros(length(u))
                        shape_coeffs = zeros(3)
                        FVM.linear_shape_function_coefficients!(shape_coeffs, u, prob, V)
                        α, β, γ = shape_coeffs
                        FVM.fvm_eqs_edge!(_du, t, α, β, γ, prob, flux_cache, V)
                        @test du == _du
                        du = zeros(length(u))
                        FVM.fvm_eqs_interior_element!(du, u, t, prob, shape_coeffs, flux_cache)
                        _du = zeros(length(u))
                        [FVM.fvm_eqs_interior_element!(_du, u, t, prob, shape_coeffs, flux_cache, V) for V in FVM.get_interior_elements(prob)]
                        @test du ≈ _du

                        du = zeros(length(u))
                        V = (21, 22, 26)
                        E = FVM.get_interior_edges(prob, V)
                        FVM.fvm_eqs_boundary_element!(du, u, t, prob, shape_coeffs, flux_cache, V)
                        _du = zeros(length(u))
                        FVM.linear_shape_function_coefficients!(shape_coeffs, u, prob, V)
                        [FVM.fvm_eqs_edge!(_du, t, (vj, j), (vjnb, jnb), shape_coeffs..., prob, flux_cache, V) for ((vj, j), (vjnb, jnb)) in E]
                        @test du == _du
                        du = zeros(length(u))
                        FVM.fvm_eqs_boundary_element!(du, u, t, prob, shape_coeffs, flux_cache)
                        _du = zeros(length(u))
                        [FVM.fvm_eqs_boundary_element!(_du, u, t, prob, shape_coeffs, flux_cache, V) for V in FVM.get_boundary_elements(prob)]
                        @test du ≈ _du

                        du = rand(length(u))
                        old_du = deepcopy(du)
                        u = rand(length(du))
                        t = rand()
                        j = 7
                        FVM.fvm_eqs_source_contribution!(du, u, t, j, prob)
                        @test du[j] ≈ old_du[j] / prob.mesh.volumes[j] + prob.reaction_function(pts[:, j]..., t, u[j], prob.reaction_parameters)

                        du = rand(length(u))
                        _du = deepcopy(du)
                        u = rand(length(du))
                        t = rand()
                        FVM.fvm_eqs_source_contribution!(du, u, t, prob)
                        [FVM.fvm_eqs_source_contribution!(_du, u, t, j, prob) for j in FVM.get_interior_or_neumann_nodes(prob)]
                        @test du == _du

                        for j in FVM.get_boundary_nodes(prob)
                            FVM.evaluate_boundary_function!(du, u, t, j, prob)
                            @test du[j] ≈ prob.boundary_conditions.functions[prob.boundary_conditions.type_map[j]](pts[:, j]..., t, u[j], prob.boundary_conditions.parameters[prob.boundary_conditions.type_map[j]])
                        end
                        j = rand(FVM.get_boundary_nodes(prob))
                        SHOW_WARNTYPE && @code_warntype FVM.evaluate_boundary_function!(du, u, t, j, prob)

                        flux_cache = DiffCache(zeros(Float64, 2), 12)
                        shape_coeffs = DiffCache(zeros(Float64, 3), 12)
                        du = rand(30)
                        u = rand(30)
                        FVM.fvm_eqs!(du, u, (prob, flux_cache, shape_coeffs), 0.0)
                        SHOW_WARNTYPE && @code_warntype FVM.fvm_eqs!(du, u, (prob, flux_cache, shape_coeffs), 0.0)

                        u = rand(30)
                        _u = deepcopy(u)
                        FVM.update_dirichlet_nodes!(u, 0.0, prob)
                        for j in FVM.get_dirichlet_nodes(prob)
                            @test u[j] ≈ pts[1, j] * 1.0 * 2.0 + _u[j]
                        end

                        jac = FVM.jacobian_sparsity(prob)
                        @test jac[:, 1].nzind == [1, 2, 6]
                        @test jac[:, 2].nzind == [1, 2, 3, 6, 7]
                        @test jac[:, 3].nzind == [2, 3, 4, 7, 8]
                        @test jac[:, 4].nzind == [3, 4, 5, 8, 9]
                        @test jac[:, 5].nzind == [4, 5, 9, 10]
                        @test jac[:, 6].nzind == [1, 2, 6, 7, 11]
                        @test jac[:, 7].nzind == [2, 3, 6, 7, 8, 11, 12]
                        @test jac[:, 8].nzind == [3, 4, 7, 8, 9, 12, 13]
                        @test jac[:, 9].nzind == [4, 5, 8, 9, 10, 13, 14]
                        @test jac[:, 10].nzind == [5, 9, 10, 14, 15]
                        @test jac[:, 11].nzind == [6, 7, 11, 12, 16]
                        @test jac[:, 12].nzind == [7, 8, 11, 12, 13, 16, 17]
                        @test jac[:, 13].nzind == [8, 9, 12, 13, 14, 17, 18]
                        @test jac[:, 14].nzind == [9, 10, 13, 14, 15, 18, 19]
                        @test jac[:, 15].nzind == [10, 14, 15, 19, 20]
                        @test jac[:, 16].nzind == [11, 12, 16, 17, 21]
                        @test jac[:, 17].nzind == [12, 13, 16, 17, 18, 21, 22]
                        @test jac[:, 18].nzind == [13, 14, 17, 18, 19, 22, 23]
                        @test jac[:, 19].nzind == [14, 15, 18, 19, 20, 23, 24]
                        @test jac[:, 20].nzind == [15, 19, 20, 24, 25]
                        @test jac[:, 21].nzind == [16, 17, 21, 22, 26]
                        @test jac[:, 22].nzind == [17, 18, 21, 22, 23, 26, 27]
                        @test jac[:, 23].nzind == [18, 19, 22, 23, 24, 27, 28]
                        @test jac[:, 24].nzind == [19, 20, 23, 24, 25, 28, 29]
                        @test jac[:, 25].nzind == [20, 24, 25, 29, 30]
                        @test jac[:, 26].nzind == [21, 22, 26, 27]
                        @test jac[:, 27].nzind == [22, 23, 26, 27, 28]
                        @test jac[:, 28].nzind == [23, 24, 27, 28, 29]
                        @test jac[:, 29].nzind == [24, 25, 28, 29, 30]
                        @test jac[:, 30].nzind == [25, 29, 30]
                        @test jac == sparse((DelaunayTriangulation.SimpleGraphs.adjacency(get_graph(get_graph(tri)))+I)[5:end, 5:end]) # 5:end because of the boundary indices
                    end
                end
            end
        end
    end
end