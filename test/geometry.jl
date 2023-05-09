using ..FiniteVolumeMethod
using Test
using LinearAlgebra
using CairoMakie
include("test_setup.jl")

## Define the test geometry 
a, b, c, d, nx, ny, tri = example_triangulation()

## Look at it 
fig = Figure()
ax = Axis(fig[1, 1])
triplot!(ax, tri)
text!(ax, get_points(tri); text=string.(each_point_index(tri)))

## Define the geometry 
for coordinate_type in (Vector{Float64}, NTuple{2,Float64}, SVector{2,Float64})
    for control_volume_storage_type_vector in (Vector{coordinate_type}, NTuple{3,coordinate_type}, SVector{3,coordinate_type})
        for control_volume_storage_type_scalar in (Vector{Float64}, NTuple{3,Float64}, SVector{3,Float64})
            for shape_function_coefficient_storage_type in (Vector{Float64}, NTuple{9,Float64}, SVector{9,Float64})
                for interior_edge_storage_type in (Vector{Int64}, NTuple{2,Int64}, SVector{2,Int64})
                    for interior_edge_pair_storage_type in (Vector{interior_edge_storage_type}, NTuple{2,interior_edge_storage_type}, SVector{2,interior_edge_storage_type})
                        geo = FVMGeometry(tri;
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
                        @test DT.compare_triangle_collections(boundary_elements, true_boundary_elements)

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
                        @test DT.compare_triangle_collections(elements, true_elements)

                        interior_edge_boundary_element_identifier = interior_info.interior_edge_boundary_element_identifier
                        true_interior_edge_boundary_element_identifier = get_interior_identifier_for_example_triangulation(interior_edge_pair_storage_type)
                        @test interior_edge_boundary_element_identifier == true_interior_edge_boundary_element_identifier

                        interior_nodes = interior_info.nodes
                        true_interior_nodes = [7, 8, 9, 12, 13, 14, 17, 18, 19, 22, 23, 24]
                        @test interior_nodes == true_interior_nodes

                        ## Test the mesh information 
                        mesh_info = FVM.get_mesh_information(geo)
                        @test mesh_info == geo.mesh_information

                        tria = mesh_info.triangulation
                        @test tria == tri

                        adjm = mesh_info.triangulation.adjacent
                        @test get_adjacent(tri) == adjm

                        adj2vm = mesh_info.triangulation.adjacent2vertex
                        @test adj2vm == get_adjacent2vertex(tri)

                        elementsm = mesh_info.triangulation.triangles
                        @test elementsm == get_triangles(tri)

                        neighboursm = mesh_info.triangulation.graph
                        @test neighboursm == get_graph(tri)

                        pointsm = mesh_info.triangulation.points
                        @test pointsm == get_points(tri)

                        total_area = mesh_info.total_area
                        @test total_area ≈ (b - a) * (d - c)

                        ## Test the element information 
                        element_info = FVM.get_element_information(geo)
                        @test element_info == geo.element_information_list

                        for elements in each_solid_triangle(tri)
                            element = element_info[elements]
                            @test element == FVM.get_element_information(geo, elements)
                            i, j, k = elements
                            p, q, r = get_point(tri, i, j, k)
                            @test DT.polygon_features([p, q, r], [1, 2, 3, 1])[1] ≈ element.area
                            @test all(element.centroid .≈ (p .+ q .+ r) ./ 3)
                            m₁ = (p .+ q) ./ 2
                            m₂ = (q .+ r) ./ 2
                            m₃ = (r .+ p) ./ 2
                            @test all(element.midpoints[1] .≈ m₁)
                            @test all(element.midpoints[2] .≈ m₂)
                            @test all(element.midpoints[3] .≈ m₃)
                            @test all(element.control_volume_edge_midpoints[1] .≈ (5 .* p .+ 5 .* q .+ 2 .* r) ./ 12) # (m₁ + c) = (p + q)/2 + (p+q+r)/3 = (5p + 5q + 2r)/6
                            @test all(element.control_volume_edge_midpoints[2] .≈ (2 .* p .+ 5 .* q .+ 5 .* r) ./ 12) # (m₁ + c) = (p + q)/2 + (p+q+r)/3 = (5p + 5q + 2r)/6
                            @test all(element.control_volume_edge_midpoints[3] .≈ (5 .* p .+ 2 .* q .+ 5 .* r) ./ 12) # (m₁ + c) = (p + q)/2 + (p+q+r)/3 = (5p + 5q + 2r)/6
                            @test element.lengths[1] ≈ norm((p .+ q) ./ 2 .- (p .+ q .+ r) ./ 3)
                            @test element.lengths[2] ≈ norm((q .+ r) ./ 2 .- (p .+ q .+ r) ./ 3)
                            @test element.lengths[3] ≈ norm((r .+ p) ./ 2 .- (p .+ q .+ r) ./ 3)
                            @test all([0 -1; 1 0] * collect(((p .+ q) ./ 2 .- (p .+ q .+ r) ./ 3) ./ element.lengths[1]) .≈ element.normals[1])
                            @test all([0 -1; 1 0] * collect(((q .+ r) ./ 2 .- (p .+ q .+ r) ./ 3) ./ element.lengths[2]) .≈ element.normals[2])
                            @test all([0 -1; 1 0] * collect(((r .+ p) ./ 2 .- (p .+ q .+ r) ./ 3) ./ element.lengths[3]) .≈ element.normals[3])
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

## Test the geometry code on an annulus
R₁ = 2.0
R₂ = 3.0
θ = LinRange(0, 2π, 25)
θ = collect(θ)
θ[end] = θ[begin]
inner = R₁
outer = R₂
x = [
    [outer .* cos.(θ)],
    [reverse(inner .* cos.(θ))]
]
y = [
    [outer .* sin.(θ)],
    [reverse(inner .* sin.(θ))]
]
boundary_nodes, points = convert_boundary_points_to_indices(x, y)
tri = triangulate(points; boundary_nodes)
refine!(tri; max_area=1e-2get_total_area(tri))
mesh = FVMGeometry(tri)

θ = LinRange(0, 2π, 500)
θ = collect(θ)
θ[end] = θ[begin]
inner = R₁
outer = R₂
x = [
    [outer .* cos.(θ)],
    [reverse(inner .* cos.(θ))]
]
y = [
    [outer .* sin.(θ)],
    [reverse(inner .* sin.(θ))]
]
boundary_nodes, points = convert_boundary_points_to_indices(x, y)
tri2 = triangulate(points; boundary_nodes)

for coordinate_type in (Vector{Float64}, NTuple{2,Float64}, SVector{2,Float64})
    for control_volume_storage_type_vector in (Vector{coordinate_type}, NTuple{3,coordinate_type}, SVector{3,coordinate_type})
        for control_volume_storage_type_scalar in (Vector{Float64}, NTuple{3,Float64}, SVector{3,Float64})
            for shape_function_coefficient_storage_type in (Vector{Float64}, NTuple{9,Float64}, SVector{9,Float64})
                for interior_edge_storage_type in (Vector{Int64}, NTuple{2,Int64}, SVector{2,Int64})
                    for interior_edge_pair_storage_type in (Vector{interior_edge_storage_type}, NTuple{2,interior_edge_storage_type}, SVector{2,interior_edge_storage_type})
                        geo = FVMGeometry(tri;
                            coordinate_type, control_volume_storage_type_vector,
                            control_volume_storage_type_scalar, shape_function_coefficient_storage_type,
                            interior_edge_storage_type, interior_edge_pair_storage_type)
                        ## Test the boundary information 
                        boundary_info = FVM.get_boundary_information(geo)
                        boundary_elements = boundary_info.boundary_elements
                        local boundary_nodes
                        boundary_nodes = boundary_info.boundary_nodes
                        true_boundary_nodes = collect(1:48)

                        edge_information = boundary_info.edge_information
                        adjacent_nodes = edge_information.adjacent_nodes
                        true_adjacent_nodes = [71, 66, 68, 72, 57, 69, 67, 59, 58, 60, 54, 52, 63, 53, 56, 64, 49, 70, 65, 50, 62, 61, 51, 55, 55, 51, 61, 62, 50, 65, 70, 49, 64, 56, 53, 63, 52, 54, 60, 58, 59, 67, 69, 57, 72, 68, 66, 71]
                        left_nodes = edge_information.left_nodes
                        true_left_nodes = boundary_nodes
                        @test left_nodes == true_left_nodes
                        right_nodes = edge_information.right_nodes
                        true_right_nodes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 1, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 25]
                        types = edge_information.types
                        true_types = [repeat([1], 24)..., repeat([2], 24)...]

                        ## Test the interior information 
                        interior_info = FVM.get_interior_information(geo)
                        @test interior_info == geo.interior_information

                        elements = interior_info.elements
                        true_elements = Set{NTuple{3,Int64}}([
                            (7, 67, 69),
                            (6, 69, 57),
                            (5, 57, 72),
                            (4, 72, 68),
                            (3, 68, 66),
                            (2, 66, 71),
                            (1, 71, 55),
                            (24, 55, 51),
                            (23, 51, 61),
                            (22, 61, 62),
                            (21, 62, 50),
                            (20, 50, 65),
                            (19, 65, 70),
                            (18, 70, 49),
                            (17, 49, 64),
                            (16, 64, 56),
                            (15, 56, 53),
                            (14, 53, 63),
                            (13, 63, 52),
                            (12, 52, 54),
                            (11, 54, 60),
                            (10, 60, 58),
                            (9, 58, 59),
                            (8, 59, 67),
                            (67, 43, 69),
                            (69, 44, 57),
                            (57, 45, 72),
                            (72, 46, 68),
                            (68, 47, 66),
                            (66, 48, 71),
                            (71, 25, 55),
                            (55, 26, 51),
                            (51, 27, 61),
                            (61, 28, 62),
                            (62, 29, 50),
                            (50, 30, 65),
                            (65, 31, 70),
                            (70, 32, 49),
                            (49, 33, 64),
                            (64, 34, 56),
                            (56, 35, 53),
                            (53, 36, 63),
                            (63, 37, 52),
                            (52, 38, 54),
                            (54, 39, 60),
                            (60, 40, 58),
                            (58, 41, 59),
                            (59, 42, 67),
                        ])

                        interior_nodes = interior_info.nodes
                        true_interior_nodes = vec([49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72])

                        ## Test the mesh information 
                        mesh_info = FVM.get_mesh_information(geo)
                        @test mesh_info == geo.mesh_information

                        tria = mesh_info.triangulation
                        @test tria == tri

                        adjm = mesh_info.triangulation.adjacent
                        @test get_adjacent(tri) == adjm

                        adj2vm = mesh_info.triangulation.adjacent2vertex
                        @test adj2vm == get_adjacent2vertex(tri)

                        elementsm = mesh_info.triangulation.triangles
                        @test elementsm == get_triangles(tri)

                        neighboursm = mesh_info.triangulation.graph
                        @test neighboursm == get_graph(tri)

                        pointsm = mesh_info.triangulation.points
                        @test pointsm == get_points(tri)

                        total_area = mesh_info.total_area
                        @test total_area ≈ π * (outer^2 - inner^2) rtol = 1e-1

                        ## Test the element information 
                        element_info = FVM.get_element_information(geo)
                        @test element_info == geo.element_information_list

                        for elements in each_solid_triangle(tri)
                            element = element_info[elements]
                            @test element == FVM.get_element_information(geo, elements)
                            i, j, k = elements
                            p, q, r = get_point(tri, i, j, k)
                            @test DT.polygon_features([p, q, r], [1, 2, 3, 1])[1] ≈ element.area
                            @test all(element.centroid .≈ (p .+ q .+ r) ./ 3)
                            m₁ = (p .+ q) ./ 2
                            m₂ = (q .+ r) ./ 2
                            m₃ = (r .+ p) ./ 2
                            @test all(element.midpoints[1] .≈ m₁)
                            @test all(element.midpoints[2] .≈ m₂)
                            @test all(element.midpoints[3] .≈ m₃)
                            @test collect(element.control_volume_edge_midpoints[1]) ≈ collect(5 .* p .+ 5 .* q .+ 2 .* r) ./ 12 # (m₁ + c) = (p + q)/2 + (p+q+r)/3 = (5p + 5q + 2r)/6
                            @test collect(element.control_volume_edge_midpoints[2]) ≈ collect(2 .* p .+ 5 .* q .+ 5 .* r) ./ 12 # (m₁ + c) = (p + q)/2 + (p+q+r)/3 = (5p + 5q + 2r)/6
                            @test collect(element.control_volume_edge_midpoints[3]) ≈ collect(5 .* p .+ 2 .* q .+ 5 .* r) ./ 12  # (m₁ + c) = (p + q)/2 + (p+q+r)/3 = (5p + 5q + 2r)/6
                            @test element.lengths[1] ≈ norm((p .+ q) ./ 2 .- (p .+ q .+ r) ./ 3)
                            @test element.lengths[2] ≈ norm((q .+ r) ./ 2 .- (p .+ q .+ r) ./ 3)
                            @test element.lengths[3] ≈ norm((r .+ p) ./ 2 .- (p .+ q .+ r) ./ 3)
                            @test all([0 -1; 1 0] * collect(((p .+ q) ./ 2 .- (p .+ q .+ r) ./ 3) ./ element.lengths[1]) .≈ element.normals[1])
                            @test all([0 -1; 1 0] * collect(((q .+ r) ./ 2 .- (p .+ q .+ r) ./ 3) ./ element.lengths[2]) .≈ element.normals[2])
                            @test all([0 -1; 1 0] * collect(((r .+ p) ./ 2 .- (p .+ q .+ r) ./ 3) ./ element.lengths[3]) .≈ element.normals[3])
                            @test sum(element.shape_function_coefficients) ≈ 1
                        end

                        ## Test the volumes 
                        volumes = geo.volumes
                        @test sum(values(volumes)) ≈ geo.mesh_information.total_area

                        ## Test the normals for the bigger problem 
                        geo = FVMGeometry(tri2;
                            coordinate_type, control_volume_storage_type_vector,
                            control_volume_storage_type_scalar, shape_function_coefficient_storage_type,
                            interior_edge_storage_type, interior_edge_pair_storage_type)
                        boundary_info = FVM.get_boundary_information(geo)
                        normal_information = boundary_info.normal_information
                        x_normals = normal_information.x_normals
                        y_normals = normal_information.y_normals
                        boundary_nodes = boundary_info.boundary_nodes
                        for i in boundary_nodes
                            p = get_point(tri2, i)
                            d1 = abs(norm(p) - outer)
                            d2 = abs(norm(p) - inner)
                            if d1 < d2 # outer 
                                x_norm = getx(p) / norm(p)
                                y_norm = gety(p) / norm(p)
                                @test x_normals[i] ≈ x_norm rtol = 1e-2 atol = 1e-2
                                @test y_normals[i] ≈ y_norm rtol = 1e-2 atol = 1e-2
                                @test sqrt(x_normals[i]^2 + y_normals[i]^2) ≈ 1.0
                            else # inner => normal is inwards 
                                x_norm = -getx(p) / norm(p)
                                y_norm = -gety(p) / norm(p)
                                @test x_normals[i] ≈ x_norm rtol = 1e-2 atol = 1e-2
                                @test y_normals[i] ≈ y_norm rtol = 1e-2 atol = 1e-2
                                @test sqrt(x_normals[i]^2 + y_normals[i]^2) ≈ 1.0
                            end
                        end
                    end
                end
            end
        end
    end
end