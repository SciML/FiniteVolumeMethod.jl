## Define the test geometry 
a, b, c, d, nx, ny, T, adj, adj2v, DG, pts, BN = example_triangulation()

## Look at it 
fig = Figure()
ax = Axis(fig[1, 1])
poly!(ax, pts, [collect(T)[i][j] for i in 1:length(T), j in 1:3], color=(:white, 0.0), strokewidth=2)
text!(ax, pts; text=string.(axes(pts, 2)))

## Define the geometry 
for coordinate_type in (Vector{Float64}, NTuple{2,Float64}, SVector{2,Float64})
    for control_volume_storage_type_vector in (Vector{coordinate_type}, NTuple{3,coordinate_type}, SVector{3,coordinate_type})
        for control_volume_storage_type_scalar in (Vector{Float64}, NTuple{3,Float64}, SVector{3,Float64})
            for shape_function_coefficient_storage_type in (Vector{Float64}, NTuple{9,Float64}, SVector{9,Float64})
                for interior_edge_storage_type in (Vector{Int64}, NTuple{2,Int64}, SVector{2,Int64})
                    for interior_edge_pair_storage_type in (Vector{interior_edge_storage_type}, NTuple{2,interior_edge_storage_type}, SVector{2,interior_edge_storage_type})
                        for i in 1:2
                            geo = i == 1 ? FVMGeometry(T, adj, adj2v, DG, pts, BN;
                                coordinate_type, control_volume_storage_type_vector,
                                control_volume_storage_type_scalar, shape_function_coefficient_storage_type,
                                interior_edge_storage_type, interior_edge_pair_storage_type) : FVMGeometry(Triangulation(T, adj, adj2v, DG, pts), BN;
                                coordinate_type, control_volume_storage_type_vector,
                                control_volume_storage_type_scalar, shape_function_coefficient_storage_type,
                                interior_edge_storage_type, interior_edge_pair_storage_type)
                            ## Test the boundary information 
                            boundary_info = FVM.get_boundary_information(geo)
                            @test boundary_info == geo.boundary_information
                            boundary_elements = boundary_info.boundary_elements
                            true_boundary_elements = Set{NTuple{3,Int64}}([(6, 1, 2), (7, 2, 3), (8, 3, 4), (9, 4, 5), (9, 5, 10),
                                (14, 10, 15), (19, 15, 20), (24, 20, 25), (29, 25, 30), (28, 24, 29), (27, 23, 28), (26, 22, 27),
                                (26, 21, 22), (21, 16, 17), (16, 11, 12), (11, 6, 7)])
                            @test DT.compare_triangle_sets(boundary_elements, true_boundary_elements)

                            boundary_nodes = boundary_info.boundary_nodes
                            true_boundary_nodes = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 29, 28, 27, 26, 21, 16, 11, 6]
                            @test boundary_nodes == true_boundary_nodes

                            edge_information = boundary_info.edge_information
                            adjacent_nodes = edge_information.adjacent_nodes
                            true_adjacent_nodes = [6, 7, 8, 9, 9, 14, 19, 24, 29, 25, 24, 23, 22, 22, 17, 12, 7, 2]
                            @test adjacent_nodes == true_adjacent_nodes
                            left_nodes = edge_information.left_nodes
                            true_left_nodes = boundary_nodes
                            @test left_nodes == true_left_nodes
                            right_nodes = edge_information.right_nodes
                            true_right_nodes = [2, 3, 4, 5, 10, 15, 20, 25, 30, 29, 28, 27, 26, 21, 16, 11, 6, 1]
                            @test right_nodes == true_right_nodes
                            types = edge_information.types
                            true_types = [1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4]
                            @test types == true_types

                            normal_information = boundary_info.normal_information
                            types = normal_information.types
                            @test types == true_types
                            x_normals = normal_information.x_normals
                            true_x_normals = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0]
                            @test x_normals == true_x_normals
                            y_normals = normal_information.y_normals
                            true_y_normals = [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                            @test y_normals == true_y_normals

                            ## Test the interior information 
                            interior_info = FVM.get_interior_information(geo)
                            @test interior_info == geo.interior_information

                            elements = interior_info.elements
                            true_elements = Set{NTuple{3,Int64}}([(6, 2, 7), (7, 3, 8), (8, 4, 9), (14, 9, 10),
                                (13, 9, 14), (13, 8, 9), (12, 8, 13), (12, 7, 8), (11, 7, 12),
                                (16, 12, 17), (17, 12, 13), (17, 13, 18), (13, 14, 18), (18, 14, 19),
                                (19, 14, 15), (21, 17, 22), (22, 17, 18), (22, 18, 23),
                                (23, 18, 19), (23, 19, 24), (24, 19, 20), (27, 22, 23),
                                (28, 23, 24), (29, 24, 25)])
                            @test DT.compare_triangle_sets(elements, true_elements)

                            interior_edge_boundary_element_identifier = interior_info.interior_edge_boundary_element_identifier
                            true_interior_edge_boundary_element_identifier = get_interior_identifier_for_example_triangulation(interior_edge_pair_storage_type)
                            @test interior_edge_boundary_element_identifier == true_interior_edge_boundary_element_identifier

                            interior_nodes = interior_info.nodes
                            true_interior_nodes = [7, 8, 9, 12, 13, 14, 17, 18, 19, 22, 23, 24]
                            @test interior_nodes == true_interior_nodes

                            ## Test the mesh information 
                            mesh_info = FVM.get_mesh_information(geo)
                            @test mesh_info == geo.mesh_information

                            adjm = mesh_info.adjacent
                            @test adj == adjm

                            adj2vm = mesh_info.adjacent2vertex
                            @test adj2vm == adj2v

                            elementsm = mesh_info.elements
                            @test elementsm == T

                            neighboursm = mesh_info.neighbours
                            @test neighboursm == DG

                            pointsm = mesh_info.points
                            @test pointsm == pts

                            total_area = mesh_info.total_area
                            @test total_area ≈ (b - a) * (d - c)

                            ## Test the element information 
                            element_info = FVM.get_element_information(geo)
                            @test element_info == geo.element_information_list

                            for elements in T
                                element = element_info[elements]
                                @test element == FVM.get_element_information(geo, elements)
                                i, j, k = elements
                                p, q, r = pts[:, i], pts[:, j], pts[:, k]
                                @test DT.area((p, q, r)) == element.area
                                @test all(element.centroid .≈ (p + q + r) / 3)
                                m₁ = (p + q) / 2
                                m₂ = (q + r) / 2
                                m₃ = (r + p) / 2
                                @test all(element.midpoints[1] .≈ m₁)
                                @test all(element.midpoints[2] .≈ m₂)
                                @test all(element.midpoints[3] .≈ m₃)
                                @test all(element.control_volume_edge_midpoints[1] .≈ (5p + 5q + 2r) / 12) # (m₁ + c) = (p + q)/2 + (p+q+r)/3 = (5p + 5q + 2r)/6
                                @test all(element.control_volume_edge_midpoints[2] .≈ (2p + 5q + 5r) / 12) # (m₁ + c) = (p + q)/2 + (p+q+r)/3 = (5p + 5q + 2r)/6
                                @test all(element.control_volume_edge_midpoints[3] .≈ (5p + 2q + 5r) / 12) # (m₁ + c) = (p + q)/2 + (p+q+r)/3 = (5p + 5q + 2r)/6
                                @test element.lengths[1] ≈ norm((p + q) / 2 - (p + q + r) / 3)
                                @test element.lengths[2] ≈ norm((q + r) / 2 - (p + q + r) / 3)
                                @test element.lengths[3] ≈ norm((r + p) / 2 - (p + q + r) / 3)
                                @test all([0 -1; 1 0] * ((p + q) / 2 - (p + q + r) / 3) / element.lengths[1] .≈ element.normals[1])
                                @test all([0 -1; 1 0] * ((q + r) / 2 - (p + q + r) / 3) / element.lengths[2] .≈ element.normals[2])
                                @test all([0 -1; 1 0] * ((r + p) / 2 - (p + q + r) / 3) / element.lengths[3] .≈ element.normals[3])
                                @test sum(element.shape_function_coefficients) ≈ 1
                            end

                            ## Test the volumes 
                            volumes = geo.volumes
                            @test sum(values(volumes)) ≈ geo.mesh_information.total_area
                        end
                    end
                end
            end
        end
    end
end