using FiniteVolumeMethod
using DelaunayTriangulation
using Test
using LinearAlgebra
using CairoMakie
using StaticArrays
using SciMLBase
using LinearSolve
using OrdinaryDiffEq
using StatsBase
using DiffEqBase
using SparseArrays
import DiffEqBase: dualgen
using BenchmarkTools
using FastGaussQuadrature
using Cubature
using PreallocationTools
using Bessels
const DT = DelaunayTriangulation
const FVM = FiniteVolumeMethod
global SHOW_WARNTYPE = false
global SAVE_FIGURE = false
const GMSH_PATH = "./gmsh-4.9.4-Windows64/gmsh.exe"

function example_triangulation()
    a, b, c, d = 0.0, 2.0, 0.0, 5.0
    nx = 5
    ny = 6
    T, adj, adj2v, DG, pts, BN = triangulate_structured(a, b, c, d, nx, ny; return_boundary_types=true)
    return a, b, c, d, nx, ny, T, adj, adj2v, DG, pts, BN
end
function get_interior_identifier_for_example_triangulation(interior_edge_pair_storage_type)
    if interior_edge_pair_storage_type == Vector{Vector{Int64}}
        true_interior_edge_boundary_element_identifier = Dict{NTuple{3,Int64},Vector{interior_edge_pair_storage_type}}(
            (21, 22, 26) => [[[21, 1], [22, 2]], [[22, 2], [26, 3]]],
            (9, 5, 10) => [[[9, 1], [5, 2]], [[10, 3], [9, 1]]],
            (1, 2, 6) => [[[2, 2], [6, 3]]],
            (19, 15, 20) => [[[19, 1], [15, 2]], [[20, 3], [19, 1]]],
            (2, 3, 7) => [[[3, 2], [7, 3]], [[7, 3], [2, 1]]],
            (6, 7, 11) => [[[6, 1], [7, 2]], [[7, 2], [11, 3]]],
            (14, 10, 15) => [[[14, 1], [10, 2]], [[15, 3], [14, 1]]],
            (4, 5, 9) => [[[5, 2], [9, 3]], [[9, 3], [4, 1]]],
            (28, 24, 29) => [[[28, 1], [24, 2]], [[24, 2], [29, 3]]],
            (24, 20, 25) => [[[24, 1], [20, 2]], [[25, 3], [24, 1]]],
            (26, 22, 27) => [[[26, 1], [22, 2]], [[22, 2], [27, 3]]],
            (27, 23, 28) => [[[27, 1], [23, 2]], [[23, 2], [28, 3]]],
            (29, 25, 30) => [[[29, 1], [25, 2]]],
            (16, 17, 21) => [[[16, 1], [17, 2]], [[17, 2], [21, 3]]],
            (3, 4, 8) => [[[4, 2], [8, 3]], [[8, 3], [3, 1]]],
            (11, 12, 16) => [[[11, 1], [12, 2]], [[12, 2], [16, 3]]]
        )
    elseif interior_edge_pair_storage_type == Vector{NTuple{2,Int64}}
        true_interior_edge_boundary_element_identifier = Dict{NTuple{3,Int64},Vector{interior_edge_pair_storage_type}}(
            (21, 22, 26) => [[(21, 1), (22, 2)], [(22, 2), (26, 3)]],
            (9, 5, 10) => [[(9, 1), (5, 2)], [(10, 3), (9, 1)]],
            (1, 2, 6) => [[(2, 2), (6, 3)]],
            (19, 15, 20) => [[(19, 1), (15, 2)], [(20, 3), (19, 1)]],
            (2, 3, 7) => [[(3, 2), (7, 3)], [(7, 3), (2, 1)]],
            (6, 7, 11) => [[(6, 1), (7, 2)], [(7, 2), (11, 3)]],
            (14, 10, 15) => [[(14, 1), (10, 2)], [(15, 3), (14, 1)]],
            (4, 5, 9) => [[(5, 2), (9, 3)], [(9, 3), (4, 1)]],
            (28, 24, 29) => [[(28, 1), (24, 2)], [(24, 2), (29, 3)]],
            (24, 20, 25) => [[(24, 1), (20, 2)], [(25, 3), (24, 1)]],
            (26, 22, 27) => [[(26, 1), (22, 2)], [(22, 2), (27, 3)]],
            (27, 23, 28) => [[(27, 1), (23, 2)], [(23, 2), (28, 3)]],
            (29, 25, 30) => [[(29, 1), (25, 2)]],
            (16, 17, 21) => [[(16, 1), (17, 2)], [(17, 2), (21, 3)]],
            (3, 4, 8) => [[(4, 2), (8, 3)], [(8, 3), (3, 1)]],
            (11, 12, 16) => [[(11, 1), (12, 2)], [(12, 2), (16, 3)]]
        )
    elseif interior_edge_pair_storage_type == NTuple{2,NTuple{2,Int64}}
        true_interior_edge_boundary_element_identifier = Dict{NTuple{3,Int64},Vector{interior_edge_pair_storage_type}}(
            (21, 22, 26) => [((21, 1), (22, 2)), ((22, 2), (26, 3))],
            (9, 5, 10) => [((9, 1), (5, 2)), ((10, 3), (9, 1))],
            (1, 2, 6) => [((2, 2), (6, 3))],
            (19, 15, 20) => [((19, 1), (15, 2)), ((20, 3), (19, 1))],
            (2, 3, 7) => [((3, 2), (7, 3)), ((7, 3), (2, 1))],
            (6, 7, 11) => [((6, 1), (7, 2)), ((7, 2), (11, 3))],
            (14, 10, 15) => [((14, 1), (10, 2)), ((15, 3), (14, 1))],
            (4, 5, 9) => [((5, 2), (9, 3)), ((9, 3), (4, 1))],
            (28, 24, 29) => [((28, 1), (24, 2)), ((24, 2), (29, 3))],
            (24, 20, 25) => [((24, 1), (20, 2)), ((25, 3), (24, 1))],
            (26, 22, 27) => [((26, 1), (22, 2)), ((22, 2), (27, 3))],
            (27, 23, 28) => [((27, 1), (23, 2)), ((23, 2), (28, 3))],
            (29, 25, 30) => [((29, 1), (25, 2))],
            (16, 17, 21) => [((16, 1), (17, 2)), ((17, 2), (21, 3))],
            (3, 4, 8) => [((4, 2), (8, 3)), ((8, 3), (3, 1))],
            (11, 12, 16) => [((11, 1), (12, 2)), ((12, 2), (16, 3))]
        )
    elseif interior_edge_pair_storage_type == NTuple{2,Vector{Int64}}
        true_interior_edge_boundary_element_identifier = Dict{NTuple{3,Int64},Vector{interior_edge_pair_storage_type}}(
            (21, 22, 26) => [([21, 1], [22, 2]), ([22, 2], [26, 3])],
            (9, 5, 10) => [([9, 1], [5, 2]), ([10, 3], [9, 1])],
            (1, 2, 6) => [([2, 2], [6, 3])],
            (19, 15, 20) => [([19, 1], [15, 2]), ([20, 3], [19, 1])],
            (2, 3, 7) => [([3, 2], [7, 3]), ([7, 3], [2, 1])],
            (6, 7, 11) => [([6, 1], [7, 2]), ([7, 2], [11, 3])],
            (14, 10, 15) => [([14, 1], [10, 2]), ([15, 3], [14, 1])],
            (4, 5, 9) => [([5, 2], [9, 3]), ([9, 3], [4, 1])],
            (28, 24, 29) => [([28, 1], [24, 2]), ([24, 2], [29, 3])],
            (24, 20, 25) => [([24, 1], [20, 2]), ([25, 3], [24, 1])],
            (26, 22, 27) => [([26, 1], [22, 2]), ([22, 2], [27, 3])],
            (27, 23, 28) => [([27, 1], [23, 2]), ([23, 2], [28, 3])],
            (29, 25, 30) => [([29, 1], [25, 2])],
            (16, 17, 21) => [([16, 1], [17, 2]), ([17, 2], [21, 3])],
            (3, 4, 8) => [([4, 2], [8, 3]), ([8, 3], [3, 1])],
            (11, 12, 16) => [([11, 1], [12, 2]), ([12, 2], [16, 3])]
        )
    elseif interior_edge_pair_storage_type == NTuple{2,SVector{2,Int64}}
        true_interior_edge_boundary_element_identifier = Dict{NTuple{3,Int64},Vector{interior_edge_pair_storage_type}}(
            (21, 22, 26) => [(@SVector[21, 1], @SVector[22, 2]), (@SVector[22, 2], @SVector[26, 3])],
            (9, 5, 10) => [(@SVector[9, 1], @SVector[5, 2]), (@SVector[10, 3], @SVector[9, 1])],
            (1, 2, 6) => [(@SVector[2, 2], @SVector[6, 3])],
            (19, 15, 20) => [(@SVector[19, 1], @SVector[15, 2]), (@SVector[20, 3], @SVector[19, 1])],
            (2, 3, 7) => [(@SVector[3, 2], @SVector[7, 3]), (@SVector[7, 3], @SVector[2, 1])],
            (6, 7, 11) => [(@SVector[6, 1], @SVector[7, 2]), (@SVector[7, 2], @SVector[11, 3])],
            (14, 10, 15) => [(@SVector[14, 1], @SVector[10, 2]), (@SVector[15, 3], @SVector[14, 1])],
            (4, 5, 9) => [(@SVector[5, 2], @SVector[9, 3]), (@SVector[9, 3], @SVector[4, 1])],
            (28, 24, 29) => [(@SVector[28, 1], @SVector[24, 2]), (@SVector[24, 2], @SVector[29, 3])],
            (24, 20, 25) => [(@SVector[24, 1], @SVector[20, 2]), (@SVector[25, 3], @SVector[24, 1])],
            (26, 22, 27) => [(@SVector[26, 1], @SVector[22, 2]), (@SVector[22, 2], @SVector[27, 3])],
            (27, 23, 28) => [(@SVector[27, 1], @SVector[23, 2]), (@SVector[23, 2], @SVector[28, 3])],
            (29, 25, 30) => [(@SVector[29, 1], @SVector[25, 2])],
            (16, 17, 21) => [(@SVector[16, 1], @SVector[17, 2]), (@SVector[17, 2], @SVector[21, 3])],
            (3, 4, 8) => [(@SVector[4, 2], @SVector[8, 3]), (@SVector[8, 3], @SVector[3, 1])],
            (11, 12, 16) => [(@SVector[11, 1], @SVector[12, 2]), (@SVector[12, 2], @SVector[16, 3])]
        )
    elseif interior_edge_pair_storage_type == SVector{2,Vector{Int64}}
        true_interior_edge_boundary_element_identifier = Dict{NTuple{3,Int64},Vector{interior_edge_pair_storage_type}}(
            (21, 22, 26) => [@SVector[[21, 1], [22, 2]], @SVector[[22, 2], [26, 3]]],
            (9, 5, 10) => [@SVector[[9, 1], [5, 2]], @SVector[[10, 3], [9, 1]]],
            (1, 2, 6) => [@SVector[[2, 2], [6, 3]]],
            (19, 15, 20) => [@SVector[[19, 1], [15, 2]], @SVector[[20, 3], [19, 1]]],
            (2, 3, 7) => [@SVector[[3, 2], [7, 3]], @SVector[[7, 3], [2, 1]]],
            (6, 7, 11) => [@SVector[[6, 1], [7, 2]], @SVector[[7, 2], [11, 3]]],
            (14, 10, 15) => [@SVector[[14, 1], [10, 2]], @SVector[[15, 3], [14, 1]]],
            (4, 5, 9) => [@SVector[[5, 2], [9, 3]], @SVector[[9, 3], [4, 1]]],
            (28, 24, 29) => [@SVector[[28, 1], [24, 2]], @SVector[[24, 2], [29, 3]]],
            (24, 20, 25) => [@SVector[[24, 1], [20, 2]], @SVector[[25, 3], [24, 1]]],
            (26, 22, 27) => [@SVector[[26, 1], [22, 2]], @SVector[[22, 2], [27, 3]]],
            (27, 23, 28) => [@SVector[[27, 1], [23, 2]], @SVector[[23, 2], [28, 3]]],
            (29, 25, 30) => [@SVector[[29, 1], [25, 2]]],
            (16, 17, 21) => [@SVector[[16, 1], [17, 2]], @SVector[[17, 2], [21, 3]]],
            (3, 4, 8) => [@SVector[[4, 2], [8, 3]], @SVector[[8, 3], [3, 1]]],
            (11, 12, 16) => [@SVector[[11, 1], [12, 2]], @SVector[[12, 2], [16, 3]]]
        )
    elseif interior_edge_pair_storage_type == SVector{2,NTuple{2,Int64}}
        true_interior_edge_boundary_element_identifier = Dict{NTuple{3,Int64},Vector{interior_edge_pair_storage_type}}(
            (21, 22, 26) => [@SVector[(21, 1), (22, 2)], @SVector[(22, 2), (26, 3)]],
            (9, 5, 10) => [@SVector[(9, 1), (5, 2)], @SVector[(10, 3), (9, 1)]],
            (1, 2, 6) => [@SVector[(2, 2), (6, 3)]],
            (19, 15, 20) => [@SVector[(19, 1), (15, 2)], @SVector[(20, 3), (19, 1)]],
            (2, 3, 7) => [@SVector[(3, 2), (7, 3)], @SVector[(7, 3), (2, 1)]],
            (6, 7, 11) => [@SVector[(6, 1), (7, 2)], @SVector[(7, 2), (11, 3)]],
            (14, 10, 15) => [@SVector[(14, 1), (10, 2)], @SVector[(15, 3), (14, 1)]],
            (4, 5, 9) => [@SVector[(5, 2), (9, 3)], @SVector[(9, 3), (4, 1)]],
            (28, 24, 29) => [@SVector[(28, 1), (24, 2)], @SVector[(24, 2), (29, 3)]],
            (24, 20, 25) => [@SVector[(24, 1), (20, 2)], @SVector[(25, 3), (24, 1)]],
            (26, 22, 27) => [@SVector[(26, 1), (22, 2)], @SVector[(22, 2), (27, 3)]],
            (27, 23, 28) => [@SVector[(27, 1), (23, 2)], @SVector[(23, 2), (28, 3)]],
            (29, 25, 30) => [@SVector[(29, 1), (25, 2)]],
            (16, 17, 21) => [@SVector[(16, 1), (17, 2)], @SVector[(17, 2), (21, 3)]],
            (3, 4, 8) => [@SVector[(4, 2), (8, 3)], @SVector[(8, 3), (3, 1)]],
            (11, 12, 16) => [@SVector[(11, 1), (12, 2)], @SVector[(12, 2), (16, 3)]]
        )
    elseif interior_edge_pair_storage_type == Vector{SVector{2,Int64}}
        true_interior_edge_boundary_element_identifier = Dict{NTuple{3,Int64},Vector{interior_edge_pair_storage_type}}(
            (21, 22, 26) => [[@SVector[21, 1], @SVector[22, 2]], [@SVector[22, 2], @SVector[26, 3]]],
            (9, 5, 10) => [[@SVector[9, 1], @SVector[5, 2]], [@SVector[10, 3], @SVector[9, 1]]],
            (1, 2, 6) => [[@SVector[2, 2], @SVector[6, 3]]],
            (19, 15, 20) => [[@SVector[19, 1], @SVector[15, 2]], [@SVector[20, 3], @SVector[19, 1]]],
            (2, 3, 7) => [[@SVector[3, 2], @SVector[7, 3]], [@SVector[7, 3], @SVector[2, 1]]],
            (6, 7, 11) => [[@SVector[6, 1], @SVector[7, 2]], [@SVector[7, 2], @SVector[11, 3]]],
            (14, 10, 15) => [[@SVector[14, 1], @SVector[10, 2]], [@SVector[15, 3], @SVector[14, 1]]],
            (4, 5, 9) => [[@SVector[5, 2], @SVector[9, 3]], [[9, 3], @SVector[4, 1]]],
            (28, 24, 29) => [[@SVector[28, 1], @SVector[24, 2]], [@SVector[24, 2], @SVector[29, 3]]],
            (24, 20, 25) => [[@SVector[24, 1], @SVector[20, 2]], [@SVector[25, 3], @SVector[24, 1]]],
            (26, 22, 27) => [[@SVector[26, 1], @SVector[22, 2]], [@SVector[22, 2], @SVector[27, 3]]],
            (27, 23, 28) => [[@SVector[27, 1], @SVector[23, 2]], [@SVector[23, 2], @SVector[28, 3]]],
            (29, 25, 30) => [[@SVector[29, 1], @SVector[25, 2]]],
            (16, 17, 21) => [[@SVector[16, 1], @SVector[17, 2]], [@SVector[17, 2], @SVector[21, 3]]],
            (3, 4, 8) => [[@SVector[4, 2], @SVector[8, 3]], [@SVector[8, 3], @SVector[3, 1]]],
            (11, 12, 16) => [[@SVector[11, 1], @SVector[12, 2]], [@SVector[12, 2], @SVector[16, 3]]]
        )
    elseif interior_edge_pair_storage_type == SVector{2,SVector{2,Int64}}
        true_interior_edge_boundary_element_identifier = Dict{NTuple{3,Int64},Vector{interior_edge_pair_storage_type}}(
            (21, 22, 26) => [@SVector[@SVector[21, 1], @SVector[22, 2]], @SVector[@SVector[22, 2], @SVector[26, 3]]],
            (9, 5, 10) => [@SVector[@SVector[9, 1], @SVector[5, 2]], @SVector[@SVector[10, 3], @SVector[9, 1]]],
            (1, 2, 6) => [@SVector[@SVector[2, 2], @SVector[6, 3]]],
            (19, 15, 20) => [@SVector[@SVector[19, 1], @SVector[15, 2]], @SVector[@SVector[20, 3], @SVector[19, 1]]],
            (2, 3, 7) => [@SVector[@SVector[3, 2], @SVector[7, 3]], @SVector[@SVector[7, 3], @SVector[2, 1]]],
            (6, 7, 11) => [@SVector[@SVector[6, 1], @SVector[7, 2]], @SVector[@SVector[7, 2], @SVector[11, 3]]],
            (14, 10, 15) => [@SVector[@SVector[14, 1], @SVector[10, 2]], @SVector[@SVector[15, 3], @SVector[14, 1]]],
            (4, 5, 9) => [@SVector[@SVector[5, 2], @SVector[9, 3]], [[9, 3], @SVector[4, 1]]],
            (28, 24, 29) => [@SVector[@SVector[28, 1], @SVector[24, 2]], @SVector[@SVector[24, 2], @SVector[29, 3]]],
            (24, 20, 25) => [@SVector[@SVector[24, 1], @SVector[20, 2]], @SVector[@SVector[25, 3], @SVector[24, 1]]],
            (26, 22, 27) => [@SVector[@SVector[26, 1], @SVector[22, 2]], @SVector[@SVector[22, 2], @SVector[27, 3]]],
            (27, 23, 28) => [@SVector[@SVector[27, 1], @SVector[23, 2]], @SVector[@SVector[23, 2], @SVector[28, 3]]],
            (29, 25, 30) => [@SVector[@SVector[29, 1], @SVector[25, 2]]],
            (16, 17, 21) => [@SVector[@SVector[16, 1], @SVector[17, 2]], @SVector[@SVector[17, 2], @SVector[21, 3]]],
            (3, 4, 8) => [@SVector[@SVector[4, 2], @SVector[8, 3]], @SVector[@SVector[8, 3], @SVector[3, 1]]],
            (11, 12, 16) => [@SVector[@SVector[11, 1], @SVector[12, 2]], @SVector[@SVector[12, 2], @SVector[16, 3]]]
        )
    end
    return true_interior_edge_boundary_element_identifier
end

###########################################################
##
## FVMGeometry 
##
###########################################################
@testset "FVMGeometry" begin
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
                            geo = FVMGeometry(T, adj, adj2v, DG, pts, BN;
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

###########################################################
##
## BoundaryConditions 
##
###########################################################
@testset "BoundaryConditions" begin
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
end

###########################################################
##
## FVMProblem
##
###########################################################
@testset "FVMProblem" begin
    ## Make sure that the flux function is being constructed correctly
    for iip_flux in (true, false)
        for q_storage in (SVector{2,Float64}, Vector{Float64}, NTuple{2,Float64})
            flux_function = nothing
            flux_parameters = nothing
            delay_function = nothing
            delay_parameters = nothing
            diffusion_function = (x, y, t, u, p) -> x * y
            diffusion_parameters = nothing
            flux_fnc = FVM.construct_flux_function(iip_flux, flux_function, delay_function, delay_parameters, diffusion_function, diffusion_parameters; q_storage)
            x, y, t, α, β, γ, p = rand(), rand(), rand(), rand(), rand(), rand(), nothing
            if iip_flux
                q = zeros(2)
                @inferred flux_fnc(q, x, y, t, α, β, γ, p)
                SHOW_WARNTYPE && @code_warntype flux_fnc(q, x, y, t, α, β, γ, p)
                u = α * x + β * y + γ
                @test all(q .≈ [-diffusion_function(x, y, t, u, diffusion_parameters) * α, -diffusion_function(x, y, t, u, diffusion_parameters) * β])
            else
                @inferred flux_fnc(x, y, t, α, β, γ, p)
                SHOW_WARNTYPE && @code_warntype flux_fnc(x, y, t, α, β, γ, p)
                q = flux_fnc(x, y, t, α, β, γ, p)
                u = α * x + β * y + γ
                @test all(q .≈ [-diffusion_function(x, y, t, u, diffusion_parameters) * α, -diffusion_function(x, y, t, u, diffusion_parameters) * β])
            end

            diffusion_function = (x, y, t, u, p) -> x * y + p
            diffusion_parameters = 3.7
            x, y, t, α, β, γ, p = rand(), rand(), rand(), rand(), rand(), rand(), nothing
            flux_fnc = FVM.construct_flux_function(iip_flux, flux_function, delay_function, delay_parameters, diffusion_function, diffusion_parameters)
            if iip_flux
                q = zeros(2)
                @inferred flux_fnc(q, x, y, t, α, β, γ, p)
                SHOW_WARNTYPE && @code_warntype flux_fnc(q, x, y, t, α, β, γ, p)
                u = α * x + β * y + γ
                @test all(q .≈ [-diffusion_function(x, y, t, u, diffusion_parameters) * α, -diffusion_function(x, y, t, u, diffusion_parameters) * β])
            else
                @inferred flux_fnc(x, y, t, α, β, γ, p)
                SHOW_WARNTYPE && @code_warntype flux_fnc(x, y, t, α, β, γ, p)
                q = flux_fnc(x, y, t, α, β, γ, p)
                u = α * x + β * y + γ
                @test all(q .≈ [-diffusion_function(x, y, t, u, diffusion_parameters) * α, -diffusion_function(x, y, t, u, diffusion_parameters) * β])
            end

            diffusion_function = (x, y, t, u, p) -> x * y
            diffusion_parameters = nothing
            delay_function = (x, y, t, u, p) -> 1 / (1 + exp(-t))
            delay_parameters = nothing
            flux_fnc = FVM.construct_flux_function(iip_flux, flux_function, delay_function, delay_parameters, diffusion_function, diffusion_parameters)
            x, y, t, α, β, γ, p = rand(), rand(), rand(), rand(), rand(), rand(), nothing
            if iip_flux
                q = zeros(2)
                @inferred flux_fnc(q, x, y, t, α, β, γ, p)
                SHOW_WARNTYPE && @code_warntype flux_fnc(q, x, y, t, α, β, γ, p)
                u = α * x + β * y + γ
                @test all(q .≈ [-delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * α,
                    -delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * β])
            else
                @inferred flux_fnc(x, y, t, α, β, γ, p)
                SHOW_WARNTYPE && @code_warntype flux_fnc(x, y, t, α, β, γ, p)
                q = flux_fnc(x, y, t, α, β, γ, p)
                u = α * x + β * y + γ
                @test all(q .≈ [-delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * α,
                    -delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * β])
            end

            diffusion_function = (x, y, t, u, p) -> x * y + p
            diffusion_parameters = 3.7
            delay_function = (x, y, t, u, p) -> 1 / (1 + exp(-t))
            delay_parameters = nothing
            flux_fnc = FVM.construct_flux_function(iip_flux, flux_function, delay_function, delay_parameters, diffusion_function, diffusion_parameters)
            x, y, t, α, β, γ, p = rand(), rand(), rand(), rand(), rand(), rand(), nothing
            if iip_flux
                q = zeros(2)
                @inferred flux_fnc(q, x, y, t, α, β, γ, p)
                SHOW_WARNTYPE && @code_warntype flux_fnc(q, x, y, t, α, β, γ, p)
                u = α * x + β * y + γ
                @test all(q .≈ [-delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * α,
                    -delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * β])
            else
                @inferred flux_fnc(x, y, t, α, β, γ, p)
                SHOW_WARNTYPE && @code_warntype flux_fnc(x, y, t, α, β, γ, p)
                q = flux_fnc(x, y, t, α, β, γ, p)
                u = α * x + β * y + γ
                @test all(q .≈ [-delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * α,
                    -delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * β])
            end

            diffusion_function = (x, y, t, u, p) -> x * y
            diffusion_parameters = nothing
            delay_function = (x, y, t, u, p) -> 1 / (1 + exp(-t)) + p[1] * p[3]
            delay_parameters = (1.0, 0.5, 2.0)
            x, y, t, α, β, γ, p = rand(), rand(), rand(), rand(), rand(), rand(), nothing
            flux_fnc = FVM.construct_flux_function(iip_flux, flux_function, delay_function, delay_parameters, diffusion_function, diffusion_parameters)
            if iip_flux
                q = zeros(2)
                @inferred flux_fnc(q, x, y, t, α, β, γ, p)
                SHOW_WARNTYPE && @code_warntype flux_fnc(q, x, y, t, α, β, γ, p)
                u = α * x + β * y + γ
                @test all(q .≈ [-delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * α,
                    -delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * β])
            else
                @inferred flux_fnc(x, y, t, α, β, γ, p)
                SHOW_WARNTYPE && @code_warntype flux_fnc(x, y, t, α, β, γ, p)
                q = flux_fnc(x, y, t, α, β, γ, p)
                u = α * x + β * y + γ
                @test all(q .≈ [-delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * α,
                    -delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * β])
            end

            diffusion_function = (x, y, t, u, p) -> x * y + p
            diffusion_parameters = 3.7
            delay_function = (x, y, t, u, p) -> 1 / (1 + exp(-t)) + p[1]
            delay_parameters = 1.5
            x, y, t, α, β, γ, p = rand(), rand(), rand(), rand(), rand(), rand(), nothing
            flux_fnc = FVM.construct_flux_function(iip_flux, flux_function, delay_function, delay_parameters, diffusion_function, diffusion_parameters)
            if iip_flux
                q = zeros(2)
                @inferred flux_fnc(q, x, y, t, α, β, γ, p)
                SHOW_WARNTYPE && @code_warntype flux_fnc(q, x, y, t, α, β, γ, p)
                u = α * x + β * y + γ
                @test all(q .≈ [-delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * α,
                    -delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * β])
            else
                @inferred flux_fnc(x, y, t, α, β, γ, p)
                SHOW_WARNTYPE && @code_warntype flux_fnc(x, y, t, α, β, γ, p)
                q = flux_fnc(x, y, t, α, β, γ, p)
                u = α * x + β * y + γ
                @test all(q .≈ [-delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * α,
                    -delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * β])
            end

            flux_function = (x, y, t, α, β, γ, p) -> x * y * t + α * β * γ + p[1]
            flux_fnc = FVM.construct_flux_function(iip_flux, flux_function, delay_function, delay_parameters, diffusion_function, diffusion_parameters)
            @test flux_fnc === flux_function
        end
    end

    ## Make sure the reaction function is being constructed correctly
    reaction_function = (x, y, t, u, p) -> u * (1 - u / p)
    reaction_parameters = 1.2
    delay_function = (x, y, t, u, p) -> x * y * t
    delay_parameters = 5.0
    reaction_fnc = FVM.construct_reaction_function(reaction_function, reaction_parameters,
        delay_function, delay_parameters)
    x, y, t, u, p = rand(), rand(), rand(), rand(), nothing
    @inferred reaction_fnc(x, y, t, u, nothing)
    SHOW_WARNTYPE && @code_warntype reaction_fnc(x, y, t, u, nothing)
    @test reaction_fnc(x, y, t, u, nothing) ≈ reaction_function(x, y, t, u, reaction_parameters) * delay_function(x, y, t, u, delay_parameters)

    delay_function = nothing
    delay_parameters = nothing
    reaction_fnc = FVM.construct_reaction_function(reaction_function, reaction_parameters,
        delay_function, delay_parameters)
    @test reaction_fnc === reaction_function

    reaction_function = nothing
    reaction_parameters = 1.2
    delay_function = (x, y, t, u, p) -> x * y * t
    delay_parameters = 5.0
    reaction_fnc = FVM.construct_reaction_function(reaction_function, reaction_parameters,
        delay_function, delay_parameters)
    x, y, t, u, p = rand(), rand(), rand(), rand(), nothing
    @inferred reaction_fnc(x, y, t, u, nothing)
    SHOW_WARNTYPE && @code_warntype reaction_fnc(x, y, t, u, nothing)
    @test reaction_fnc(x, y, t, u, nothing) ≈ 0.0
    @test reaction_fnc(x, y, t, Float32(0), nothing) === 0.0f0

    reaction_function = nothing
    reaction_parameters = nothing
    delay_function = nothing
    delay_parameters = nothing
    reaction_fnc = FVM.construct_reaction_function(reaction_function, reaction_parameters,
        delay_function, delay_parameters)
    x, y, t, u, p = rand(), rand(), rand(), rand(), nothing
    @inferred reaction_fnc(x, y, t, u, nothing)
    SHOW_WARNTYPE && @code_warntype reaction_fnc(x, y, t, u, nothing)
    @test reaction_fnc(x, y, t, u, nothing) ≈ 0.0
    @test reaction_fnc(x, y, t, Float32(0), nothing) === 0.0f0

    ## Now check that the object is constructed correctly 
    a, b, c, d, nx, ny, T, adj, adj2v, DG, pts, BN = example_triangulation()
    for coordinate_type in (NTuple{2,Float64}, SVector{2,Float64})
        for control_volume_storage_type_vector in (Vector{coordinate_type}, SVector{3,coordinate_type})
            for control_volume_storage_type_scalar in (Vector{Float64}, SVector{3,Float64})
                for shape_function_coefficient_storage_type in (Vector{Float64}, NTuple{9,Float64})
                    for interior_edge_storage_type in (Vector{Int64}, NTuple{2,Int64},)
                        for interior_edge_pair_storage_type in (Vector{interior_edge_storage_type}, NTuple{2,interior_edge_storage_type})
                            geo = FVMGeometry(T, adj, adj2v, DG, pts, BN;
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
                            boundary_node_vector = BN
                            params = ((1.0, 2.0), nothing, nothing, nothing)
                            BCs = FVM.BoundaryConditions(geo, functions, types, boundary_node_vector; params)
                            iip_flux = true
                            flux_function = (q, x, y, t, α, β, γ, p) -> (q[1] = x * y * t; q[2] = t; nothing)
                            initial_condition = zeros(20)
                            final_time = 5.0
                            prob = FVMProblem(geo, BCs; iip_flux, flux_function, initial_condition, final_time)
                            @test isinplace(prob) == iip_flux
                            @test prob.boundary_conditions == BCs
                            @test prob.flux_function == flux_function
                            @test prob.initial_condition == initial_condition
                            @test prob.mesh == geo == FVM.get_mesh(prob)
                            @test prob.reaction_parameters === nothing
                            @test prob.final_time == final_time
                            @test prob.flux_parameters === nothing
                            @test prob.initial_time == 0.0
                            @test prob.reaction_function(rand(), rand(), rand(), rand(), nothing) == 0.0
                            @inferred prob.reaction_function(rand(), rand(), rand(), rand(), nothing)
                            SHOW_WARNTYPE && @code_warntype prob.reaction_function(rand(), rand(), rand(), rand(), nothing)
                            @test prob.reaction_function(rand(), rand(), rand(), rand(Float32), nothing) == 0.0f0
                            @test prob.steady == false

                            flux_function = (q, x, y, t, α, β, γ, p) -> (q[1] = x * y * t; q[2] = p; nothing)
                            flux_parameters = 3.81
                            prob = FVMProblem(geo, BCs; iip_flux, flux_function, initial_condition, final_time, flux_parameters)
                            @test prob.flux_function == flux_function
                            @test prob.flux_parameters == flux_parameters
                            q = zeros(2)
                            x, y, t, α, β, γ = rand(6)
                            prob.flux_function(q, x, y, t, α, β, γ, flux_parameters)
                            @test q ≈ [x * y * t, flux_parameters]
                            @test prob.boundary_conditions == BCs == FVM.get_boundary_conditions(prob)
                            @test prob.flux_function == flux_function
                            @test prob.initial_condition == initial_condition
                            @test prob.mesh == geo
                            @test prob.reaction_parameters === nothing
                            @test prob.final_time == final_time
                            @test prob.flux_parameters === flux_parameters
                            @test prob.initial_time == 0.0

                            diffusion_function = (x, y, t, u, p) -> u^2 + p[1] * p[2]
                            diffusion_parameters = (2.0, 3.0)
                            reaction_function = (x, y, t, u, p) -> u * (1 - u / p[1])
                            reaction_parameters = [5.0]
                            prob = FVMProblem(geo, BCs; iip_flux, diffusion_function, diffusion_parameters, initial_condition, reaction_function, reaction_parameters, final_time, initial_time=3.71)
                            @test isinplace(prob) == iip_flux
                            @test prob.boundary_conditions == BCs == FVM.get_boundary_conditions(prob)
                            q1 = zeros(2)
                            x, y, t, α, β, γ = rand(6)
                            prob.flux_function(q1, x, y, t, α, β, γ, nothing)
                            u = α * x + β * y + γ
                            q2 = [-diffusion_function(x, y, t, u, diffusion_parameters) * α, -diffusion_function(x, y, t, u, diffusion_parameters) * β]
                            @test q1 ≈ q2
                            @test prob.reaction_function(x, y, t, u, prob.reaction_parameters) == reaction_function(x, y, t, u, reaction_parameters)
                            @inferred prob.reaction_function(x, y, t, u, prob.reaction_parameters)
                            SHOW_WARNTYPE && @code_warntype prob.reaction_function(x, y, t, u, prob.reaction_parameters)
                            @test prob.boundary_conditions == BCs
                            @test prob.initial_condition == initial_condition
                            @test prob.mesh == geo
                            @test prob.reaction_parameters === reaction_parameters
                            @test prob.final_time == final_time
                            @test prob.flux_parameters === nothing
                            @test prob.initial_time == 3.71

                            delay_function = (x, y, t, u, p) -> 1 / (1 + exp(-t * p[1]))
                            delay_parameters = [2.371]
                            prob = FVMProblem(geo, BCs; iip_flux, diffusion_function, diffusion_parameters, initial_condition, delay_function, delay_parameters, reaction_function, reaction_parameters, final_time, initial_time=3.71)
                            @test isinplace(prob) == iip_flux
                            @test prob.boundary_conditions == BCs
                            q1 = zeros(2)
                            x, y, t, α, β, γ = rand(6)
                            prob.flux_function(q1, x, y, t, α, β, γ, nothing)
                            SHOW_WARNTYPE && @code_warntype prob.flux_function(q1, x, y, t, α, β, γ, nothing)
                            u = α * x + β * y + γ
                            q2 = [-delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * α, -delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * β]
                            @test q1 ≈ q2
                            @test prob.reaction_function(x, y, t, u, prob.reaction_parameters) ≈ delay_function(x, y, t, u, delay_parameters) * reaction_function(x, y, t, u, reaction_parameters)
                            @test prob.initial_condition == initial_condition
                            @test prob.mesh == geo
                            @test prob.reaction_parameters === reaction_parameters
                            @test prob.final_time == final_time
                            @test prob.flux_parameters === nothing
                            @test prob.initial_time == 3.71
                            @test prob.steady == false

                            ## Test some of the getters 
                            for V in T
                                @test FVM.gets(prob, V) == prob.mesh.element_information_list[V].shape_function_coefficients
                                for i in 1:9
                                    @test FVM.gets(prob, V, i) == prob.mesh.element_information_list[V].shape_function_coefficients[i]
                                end
                                @test FVM.get_midpoints(prob, V) == prob.mesh.element_information_list[V].midpoints
                                for i in 1:3
                                    @test FVM.get_midpoints(prob, V, i) == prob.mesh.element_information_list[V].midpoints[i]
                                end
                                @test FVM.get_normals(prob, V) == prob.mesh.element_information_list[V].normals
                                for i in 1:3
                                    @test FVM.get_normals(prob, V, i) == prob.mesh.element_information_list[V].normals[i]
                                end
                                @test FVM.get_lengths(prob, V) == prob.mesh.element_information_list[V].lengths
                                for i in 1:3
                                    @test FVM.get_lengths(prob, V, i) == prob.mesh.element_information_list[V].lengths[i]
                                end
                            end
                            x, y, t, α, β, γ = rand(6)
                            flux_cache = zeros(2)
                            FVM.get_flux!(flux_cache, prob, x, y, t, α, β, γ)
                            flux_cache_2 = zeros(2)
                            prob.flux_function(flux_cache_2, x, y, t, α, β, γ, prob.flux_parameters)
                            @test flux_cache ≈ flux_cache_2
                            prob = FVMProblem(geo, BCs; iip_flux=false, diffusion_function, diffusion_parameters, initial_condition, delay_function, delay_parameters, reaction_function, reaction_parameters, final_time, initial_time=3.71)
                            flux_cache = FVM.get_flux(prob, x, y, t, α, β, γ)
                            @test flux_cache ≈ flux_cache_2
                            SHOW_WARNTYPE && @code_warntype FVM.get_flux(prob, x, y, t, α, β, γ)
                            @test FVM.get_interior_or_neumann_nodes(prob) == prob.boundary_conditions.interior_or_neumann_nodes
                            for j ∈ FVM.get_interior_or_neumann_nodes(prob)
                                @test FVM.is_interior_or_neumann_node(prob, j)
                            end
                            j = rand(Int64, 500)
                            setdiff!(j, FVM.get_interior_or_neumann_nodes(prob))
                            for j ∈ j
                                @test !FVM.is_interior_or_neumann_node(prob, j)
                            end
                            @test FVM.get_interior_elements(prob) == prob.mesh.interior_information.elements
                            true_interior_edge_boundary_element_identifier = get_interior_identifier_for_example_triangulation(interior_edge_pair_storage_type)
                            for (V, E) in true_interior_edge_boundary_element_identifier
                                @test FVM.get_interior_edges(prob, V) == E
                            end
                            @test FVM.get_boundary_elements(prob) == prob.mesh.boundary_information.boundary_elements
                            for j in axes(pts, 2)
                                @test get_point(prob, j) == Tuple(pts[:, j])
                                @test FVM.get_volumes(prob, j) == prob.mesh.volumes[j]
                            end
                            x, y, t, u = rand(4)
                            @test FVM.get_reaction(prob, x, y, t, u) == prob.reaction_function(x, y, t, u, prob.reaction_parameters)
                            @inferred FVM.get_reaction(prob, x, y, t, u)
                            SHOW_WARNTYPE && @code_warntype FVM.get_reaction(prob, x, y, t, u)
                            @test FVM.get_dirichlet_nodes(prob) == prob.boundary_conditions.dirichlet_nodes
                            @test FVM.get_neumann_nodes(prob) == prob.boundary_conditions.neumann_nodes
                            @test FVM.get_dudt_nodes(prob) == prob.boundary_conditions.dudt_nodes
                            @test FVM.get_boundary_nodes(prob) == prob.mesh.boundary_information.boundary_nodes
                            for i in eachindex(prob.boundary_conditions.parameters)
                                @test FVM.get_boundary_function_parameters(prob, i) == prob.boundary_conditions.parameters[i]
                            end
                            for j in FVM.get_boundary_nodes(prob)
                                @test FVM.map_node_to_segment(prob, j) == prob.boundary_conditions.type_map[j]
                            end
                            x, y, t, u = rand(4)
                            @test FVM.evaluate_boundary_function(prob, 1, x, y, t, u) ≈ prob.boundary_conditions.functions[1](x, y, t, u, FVM.get_boundary_function_parameters(prob, 1))
                            @test FVM.evaluate_boundary_function(prob, 2, x, y, t, u) ≈ prob.boundary_conditions.functions[2](x, y, t, u, FVM.get_boundary_function_parameters(prob, 2))
                            @test FVM.evaluate_boundary_function(prob, 3, x, y, t, u) ≈ prob.boundary_conditions.functions[3](x, y, t, u, FVM.get_boundary_function_parameters(prob, 3))
                            @test FVM.evaluate_boundary_function(prob, 4, x, y, t, u) ≈ prob.boundary_conditions.functions[4](x, y, t, u, FVM.get_boundary_function_parameters(prob, 4))
                            @test FVM.get_neighbours(prob) == DG
                            @test FVM.get_initial_condition(prob) == prob.initial_condition
                            @test FVM.get_initial_time(prob) == prob.initial_time
                            @test FVM.get_final_time(prob) == prob.final_time
                            @test FVM.get_time_span(prob) == (prob.initial_time, prob.final_time)
                            @test FVM.get_points(prob) == pts
                            @test num_points(prob) == size(pts, 2)
                            @test FVM.num_boundary_edges(prob) == length(FVM.get_boundary_nodes(prob))
                        end
                    end
                end
            end
        end
    end
end

###########################################################
##
## FVM Equations 
##
###########################################################
@testset "FVMEquations" begin
    ## Define the example problem 
    a, b, c, d, nx, ny, T, adj, adj2v, DG, pts, BN = example_triangulation()
    for coordinate_type in (NTuple{2,Float64}, SVector{2,Float64})
        for control_volume_storage_type_vector in (Vector{coordinate_type}, SVector{3,coordinate_type})
            for control_volume_storage_type_scalar in (Vector{Float64}, SVector{3,Float64})
                for shape_function_coefficient_storage_type in (Vector{Float64}, NTuple{9,Float64})
                    for interior_edge_storage_type in (Vector{Int64}, NTuple{2,Int64},)
                        for interior_edge_pair_storage_type in (Vector{interior_edge_storage_type}, NTuple{2,interior_edge_storage_type})
                            geo = FVMGeometry(
                                T,
                                adj,
                                adj2v,
                                DG,
                                pts,
                                BN;
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
                            boundary_node_vector = BN
                            params = ((1.0, 2.0), nothing, nothing, nothing)
                            BCs = FVM.BoundaryConditions(geo, functions, types, boundary_node_vector; params)
                            iip_flux = true
                            flux_function = (q, x, y, t, α, β, γ, p) -> (q[1] = x * y * t; q[2] = t; nothing)
                            initial_condition = zeros(20)
                            final_time = 5.0
                            prob = FVMProblem(geo, BCs; iip_flux, flux_function, initial_condition, final_time)

                            ## Now do some testing 
                            shape_cache = zeros(3)
                            u = rand(size(pts, 2))
                            for V in T
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

                            flux_cache = dualcache(zeros(Float64, 2), 12)
                            shape_coeffs = dualcache(zeros(Float64, 3), 12)
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
                            @test nnz(jac) == 2(length(edges(DG)) - FVM.num_boundary_edges(prob)) + FVM.num_points(prob)
                            @test jac == sparse((DelaunayTriangulation.adjacency(DelaunayTriangulation.graph(DG))+I)[2:end, 2:end]) # 2:end because of the BoundaryIndex
                        end
                    end
                end
            end
        end
    end
end

###########################################################
##
## Example I: Diffusion equation on a square plate 
##
###########################################################
@testset "Diffusion equation on a square plate" begin
    ## Step 1: Generate the mesh 
    a, b, c, d = 0.0, 2.0, 0.0, 2.0
    n = 500
    x₁ = LinRange(a, b, n)
    x₂ = LinRange(b, b, n)
    x₃ = LinRange(b, a, n)
    x₄ = LinRange(a, a, n)
    y₁ = LinRange(c, c, n)
    y₂ = LinRange(c, d, n)
    y₃ = LinRange(d, d, n)
    y₄ = LinRange(d, c, n)
    x = reduce(vcat, [x₁, x₂, x₃, x₄])
    y = reduce(vcat, [y₁, y₂, y₃, y₄])
    xy = [[x[i], y[i]] for i in eachindex(x)]
    unique!(xy)
    x = getx.(xy)
    y = gety.(xy)
    r = 0.03
    T, adj, adj2v, DG, points, BN = generate_mesh(x, y, r; gmsh_path=GMSH_PATH)
    mesh = FVMGeometry(T, adj, adj2v, DG, points, BN)

    ## Step 2: Define the boundary conditions 
    bc = ((x, y, t, u::T, p) where {T}) -> zero(T)
    type = :Dirichlet # or :D or :dirichlet or "D" or "Dirichlet"
    BCs = BoundaryConditions(mesh, bc, type, BN)

    ## Step 3: Define the actual PDE 
    f = (x, y) -> y ≤ 1.0 ? 50.0 : 0.0 # initial condition 
    D = (x, y, t, u, p) -> 1 / 9 # You could also define flux = (q, x, y, t, α, β, γ, p) -> (q[1] = -α/9; q[2] = -β/9)
    R = ((x, y, t, u::T, p) where {T}) -> zero(T)
    u₀ = @views f.(points[1, :], points[2, :])
    iip_flux = true
    final_time = 0.5
    prob = FVMProblem(mesh, BCs; iip_flux,
        diffusion_function=D, reaction_function=R,
        initial_condition=u₀, final_time, q_storage=Vector{Float64})

    ## Step 4: Solve
    alg = TRBDF2(linsolve=KLUFactorization(), autodiff=true)
    sol = solve(prob, alg; specialization=SciMLBase.FullSpecialize, saveat=0.05)

    ## Step 5: Visualisation 
    pt_mat = Matrix(points')
    T_mat = [collect(T)[i][j] for i in 1:length(T), j in 1:3]
    fig = Figure(resolution=(2068.72f0, 686.64f0), fontsize=38)
    ax = Axis(fig[1, 1], width=600, height=600)
    xlims!(ax, a, b)
    ylims!(ax, c, d)
    mesh!(ax, pt_mat, T_mat, color=sol.u[1], colorrange=(0, 50), colormap=:matter)
    ax = Axis(fig[1, 2], width=600, height=600)
    xlims!(ax, a, b)
    ylims!(ax, c, d)
    mesh!(ax, pt_mat, T_mat, color=sol.u[6], colorrange=(0, 50), colormap=:matter)
    ax = Axis(fig[1, 3], width=600, height=600)
    xlims!(ax, a, b)
    ylims!(ax, c, d)
    mesh!(ax, pt_mat, T_mat, color=sol.u[11], colorrange=(0, 50), colormap=:matter)
    SAVE_FIGURE && save("figures/heat_equation_test.png", fig)

    ## Step 6: Define the exact solution for comparison later 
    function diffusion_equation_on_a_square_plate_exact_solution(x, y, t, N, M)
        u_exact = zeros(length(x))
        for j in eachindex(x)
            if t == 0.0
                if y[j] ≤ 1.0
                    u_exact[j] = 50.0
                else
                    u_exact[j] = 0.0
                end
            else
                u_exact[j] = 0.0
                for m = 1:M
                    for n = 1:N
                        u_exact[j] += 200 / π^2 * (1 + (-1)^(m + 1)) * (1 - cos(n * π / 2)) / (m * n) * sin(m * π * x[j] / 2) * sin(n * π * y[j] / 2) * exp(-π^2 / 36 * (m^2 + n^2) * t)
                    end
                end
            end
        end
        return u_exact
    end

    ## Step 7: Compare the results
    sol = solve(prob, alg; saveat=0.1)
    all_errs = [Float64[] for _ in eachindex(sol)]
    u_exact = [diffusion_equation_on_a_square_plate_exact_solution(points[1, :], points[2, :], τ, 200, 200) for τ in sol.t]
    u_fvm = reduce(hcat, sol.u)
    u_exact = reduce(hcat, u_exact)
    errs = reduce(hcat, [100abs.(u - û) / maximum(abs.(u)) for (u, û) in zip(eachcol(u_exact), eachcol(u_fvm))])
    @test all(<(0.15), mean.(eachcol(errs)))
    @test all(<(0.15), median.(eachcol(errs)))
    @test mean(errs) < 0.1
    @test median(errs) < 0.1

    ## Step 8: Visualise the comparison 
    fig = Figure(fontsize=42, resolution=(3469.8997f0, 1466.396f0))
    ax = Axis(fig[1, 1], width=600, height=600, title=L"(a):$ $ Exact solution, $t = %$(sol.t[1])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_exact[:, 1], colorrange=(0, 0.5), colormap=:matter)
    ax = Axis(fig[1, 2], width=600, height=600, title=L"(b):$ $ Exact solution, $t = %$(sol.t[2])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_exact[:, 2], colorrange=(0, 0.5), colormap=:matter)
    ax = Axis(fig[1, 3], width=600, height=600, title=L"(c):$ $ Exact solution, $t = %$(sol.t[3])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_exact[:, 3], colorrange=(0, 0.5), colormap=:matter)
    ax = Axis(fig[1, 4], width=600, height=600, title=L"(d):$ $ Exact solution, $t = %$(sol.t[4])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_exact[:, 4], colorrange=(0, 0.5), colormap=:matter)
    ax = Axis(fig[1, 5], width=600, height=600, title=L"(e):$ $ Exact solution, $t = %$(sol.t[5])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_exact[:, 5], colorrange=(0, 0.5), colormap=:matter)
    ax = Axis(fig[2, 1], width=600, height=600, title=L"(f):$ $ Numerical solution, $t = %$(sol.t[1])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 1], colorrange=(0, 0.5), colormap=:matter)
    ax = Axis(fig[2, 2], width=600, height=600, title=L"(g):$ $ Numerical solution, $t = %$(sol.t[2])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 2], colorrange=(0, 0.5), colormap=:matter)
    ax = Axis(fig[2, 3], width=600, height=600, title=L"(h):$ $ Numerical solution, $t = %$(sol.t[3])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 3], colorrange=(0, 0.5), colormap=:matter)
    ax = Axis(fig[2, 4], width=600, height=600, title=L"(i):$ $ Numerical solution, $t = %$(sol.t[4])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 4], colorrange=(0, 0.5), colormap=:matter)
    ax = Axis(fig[2, 5], width=600, height=600, title=L"(j):$ $ Numerical solution, $t = %$(sol.t[5])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 5], colorrange=(0, 0.5), colormap=:matter)
    SAVE_FIGURE && save("figures/heat_equation_test_error.png", fig)
end

###########################################################
##
## Example II: Diffusion equation on a wedge with mixed BCs
##
###########################################################
@testset "Diffusion equation on a wedge with mixed BCs" begin
    ## Step 1: Generate the mesh 
    n = 500
    α = π / 4

    # The bottom edge 
    r₁ = LinRange(0, 1, n)
    θ₁ = LinRange(0, 0, n)
    x₁ = @. r₁ * cos(θ₁)
    y₁ = @. r₁ * sin(θ₁)

    # Arc 
    r₂ = LinRange(1, 1, n)
    θ₂ = LinRange(0, α, n)
    x₂ = @. r₂ * cos(θ₂)
    y₂ = @. r₂ * sin(θ₂)

    # Upper edge 
    r₃ = LinRange(1, 0, n)
    θ₃ = LinRange(α, α, n)
    x₃ = @. r₃ * cos(θ₃)
    y₃ = @. r₃ * sin(θ₃)

    # Combine and create the mesh 
    x = [x₁, x₂, x₃]
    y = [y₁, y₂, y₃]
    r = 0.01
    T, adj, adj2v, DG, points, BN = generate_mesh(x, y, r; gmsh_path=GMSH_PATH)
    mesh = FVMGeometry(T, adj, adj2v, DG, points, BN)

    ## Step 2: Define the boundary conditions 
    lower_bc = ((x, y, t, u::T, p) where {T}) -> zero(T)
    arc_bc = ((x, y, t, u::T, p) where {T}) -> zero(T)
    upper_bc = ((x, y, t, u::T, p) where {T}) -> zero(T)
    types = (:N, :D, :N)
    boundary_functions = (lower_bc, arc_bc, upper_bc)
    BCs = BoundaryConditions(mesh, boundary_functions, types, BN)

    ## Step 3: Define the actual PDE  
    f = (x, y) -> 1 - sqrt(x^2 + y^2)
    D = ((x, y, t, u::T, p) where {T}) -> one(T)
    u₀ = f.(points[1, :], points[2, :])
    final_time = 0.1 # Do not need iip_flux = true or R(x, y, t, u, p) = 0, these are defaults 
    prob = FVMProblem(mesh, BCs; diffusion_function=D, initial_condition=u₀, final_time)

    flux = (q, x, y, t, α, β, γ, p) -> (q[1] = -α; q[2] = -β; nothing)
    prob2 = FVMProblem(mesh, BCs; diffusion_function=D, initial_condition=u₀, final_time)

    ## Step 4: Solve
    alg = Rosenbrock23(linsolve=UMFPACKFactorization())
    sol = solve(prob, alg; saveat=0.025)
    sol2 = solve(prob2, alg; saveat=0.025)

    ## Step 5: Visualisation 
    pt_mat = Matrix(points')
    T_mat = [collect(T)[i][j] for i in 1:length(T), j in 1:3]
    fig = Figure(resolution=(2131.8438f0, 684.27f0), fontsize=38)
    ax = Axis(fig[1, 1], width=600, height=600)
    mesh!(ax, pt_mat, T_mat, color=sol.u[1], colorrange=(0, 0.5), colormap=:matter)
    ax = Axis(fig[1, 2], width=600, height=600)
    mesh!(ax, pt_mat, T_mat, color=sol.u[3], colorrange=(0, 0.5), colormap=:matter)
    ax = Axis(fig[1, 3], width=600, height=600)
    mesh!(ax, pt_mat, T_mat, color=sol.u[5], colorrange=(0, 0.5), colormap=:matter)
    SAVE_FIGURE && save("figures/diffusion_equation_wedge_test.png", fig)

    ## Step 6: Define the exact solution for comparison later 
    function diffusion_equation_on_a_wedge_exact_solution(x, y, t, α, N, M)
        f = (r, θ) -> 1.0 - r
        ## Compute the ζ: ζ[m, n+1] is the mth zero of the (nπ/α)th order Bessel function of the first kind 
        ζ = zeros(M, N + 2)
        for n in 0:(N+1)
            order = n * π / α
            @views ζ[:, n+1] .= approx_besselroots(order, M)
        end
        A = zeros(M, N + 1) # A[m, n+1] is the coefficient Aₙₘ
        for n in 0:N
            order = n * π / α
            for m in 1:M
                integrand = rθ -> f(rθ[2], rθ[1]) * besselj(order, ζ[m, n+1] * rθ[2]) * cos(order * rθ[1]) * rθ[2]
                A[m, n+1] = 4.0 / (α * besselj(order + 1, ζ[m, n+1])^2) * hcubature(integrand, [0.0, 0.0], [α, 1.0]; abstol=1e-8)[1]
            end
        end
        r = @. sqrt(x^2 + y^2)
        θ = @. atan(y, x)
        u_exact = zeros(length(x))
        for i in 1:length(x)
            for m = 1:M
                u_exact[i] = u_exact[i] + 0.5 * A[m, 1] * exp(-ζ[m, 1]^2 * t) * besselj(0.0, ζ[m, 1] * r[i])
            end
            for n = 1:N
                order = n * π / α
                for m = 1:M
                    u_exact[i] = u_exact[i] + A[m, n+1] * exp(-ζ[m, n+1]^2 * t) * besselj(order, ζ[m, n+1] * r[i]) * cos(order * θ[i])
                end
            end
        end
        return u_exact
    end

    ## Step 7: Compare the results
    all_errs = [Float64[] for _ in eachindex(sol)]
    u_exact = [diffusion_equation_on_a_wedge_exact_solution(points[1, :], points[2, :], τ, α, 22, 24) for τ in sol.t]
    u_fvm = reduce(hcat, sol.u)
    u_exact = reduce(hcat, u_exact)
    _u_exact = deepcopy(u_exact)
    errs = reduce(hcat, [100abs.(u - û) / maximum(abs.(u)) for (u, û) in zip(eachcol(u_exact), eachcol(u_fvm))])
    @test all(<(0.3), mean.(eachcol(errs)))
    @test all(<(0.15), median.(eachcol(errs)))
    @test mean(errs) < 0.15
    @test median(errs) < 0.1

    all_errs2 = [Float64[] for _ in eachindex(sol2)]
    u_exact2 = [diffusion_equation_on_a_wedge_exact_solution(points[1, :], points[2, :], τ, α, 22, 24) for τ in sol2.t]
    u_fvm2 = reduce(hcat, sol2.u)
    u_exact2 = reduce(hcat, u_exact2)
    errs2 = reduce(hcat, [100abs.(u - û) / maximum(abs.(u)) for (u, û) in zip(eachcol(u_exact2), eachcol(u_fvm2))])
    @test errs == errs2
    @test u_fvm2 == u_fvm

    ## Step 8: Visualise the comparison 
    fig = Figure(fontsize=42, resolution=(3586.6597f0, 1466.396f0))
    ax = Axis(fig[1, 1], width=600, height=600, title=L"(a):$ $ Exact solution, $t = %$(sol.t[1])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=_u_exact[:, 1], colorrange=(0, 0.5), colormap=:matter)
    ax = Axis(fig[1, 2], width=600, height=600, title=L"(b):$ $ Exact solution, $t = %$(sol.t[2])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=_u_exact[:, 2], colorrange=(0, 0.5), colormap=:matter)
    ax = Axis(fig[1, 3], width=600, height=600, title=L"(c):$ $ Exact solution, $t = %$(sol.t[3])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=_u_exact[:, 3], colorrange=(0, 0.5), colormap=:matter)
    ax = Axis(fig[1, 4], width=600, height=600, title=L"(d):$ $ Exact solution, $t = %$(sol.t[4])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=_u_exact[:, 4], colorrange=(0, 0.5), colormap=:matter)
    ax = Axis(fig[1, 5], width=600, height=600, title=L"(e):$ $ Exact solution, $t = %$(sol.t[5])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=_u_exact[:, 5], colorrange=(0, 0.5), colormap=:matter)
    ax = Axis(fig[2, 1], width=600, height=600, title=L"(f):$ $ Numerical solution, $t = %$(sol.t[1])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=sol.u[1], colorrange=(0, 0.5), colormap=:matter)
    ax = Axis(fig[2, 2], width=600, height=600, title=L"(g):$ $ Numerical solution, $t = %$(sol.t[2])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=sol.u[2], colorrange=(0, 0.5), colormap=:matter)
    ax = Axis(fig[2, 3], width=600, height=600, title=L"(h):$ $ Numerical solution, $t = %$(sol.t[3])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=sol.u[3], colorrange=(0, 0.5), colormap=:matter)
    ax = Axis(fig[2, 4], width=600, height=600, title=L"(i):$ $ Numerical solution, $t = %$(sol.t[4])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=sol.u[4], colorrange=(0, 0.5), colormap=:matter)
    ax = Axis(fig[2, 5], width=600, height=600, title=L"(j):$ $ Numerical solution, $t = %$(sol.t[5])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=sol.u[5], colorrange=(0, 0.5), colormap=:matter)
    SAVE_FIGURE && save("figures/heat_equation_wedge_test_error.png", fig)
end

###########################################################
##
## Example III: Reaction-diffusion equation with a time-dependent Dirichlet boundary condition on a disk 
##
###########################################################
@testset "Reaction-diffusion equation with a dudt BC on a disk" begin
    ## Step 1: Generate the mesh 
    n = 500
    r = LinRange(1, 1, 1000)
    θ = LinRange(0, 2π, 1000)
    x = @. r * cos(θ)
    y = @. r * sin(θ)
    r = 0.05
    T, adj, adj2v, DG, points, BN = generate_mesh(x, y, r; gmsh_path=GMSH_PATH)
    mesh = FVMGeometry(T, adj, adj2v, DG, points, BN)

    ## Step 2: Define the boundary conditions 
    bc = (x, y, t, u, p) -> u
    types = :dudt
    BCs = BoundaryConditions(mesh, bc, types, BN)

    ## Step 3: Define the actual PDE  
    f = (x, y) -> sqrt(besseli(0.0, sqrt(2) * sqrt(x^2 + y^2)))
    D = (x, y, t, u, p) -> u
    R = (x, y, t, u, p) -> u * (1 - u)
    u₀ = [f(points[:, i]...) for i in axes(points, 2)]
    final_time = 0.10
    prob = FVMProblem(mesh, BCs; diffusion_function=D, reaction_function=R, initial_condition=u₀, final_time)

    ## Step 4: Solve
    alg = FBDF(linsolve=UMFPACKFactorization())
    sol = solve(prob, alg; saveat=0.025)

    ## Step 5: Visualisation 
    pt_mat = Matrix(points')
    T_mat = [collect(T)[i][j] for i in 1:length(T), j in 1:3]
    fig = Figure(resolution=(2131.8438f0, 684.27f0), fontsize=38)
    ax = Axis(fig[1, 1], width=600, height=600)
    mesh!(ax, pt_mat, T_mat, color=sol.u[1], colorrange=(1, 1.1), colormap=:matter)
    ax = Axis(fig[1, 2], width=600, height=600)
    mesh!(ax, pt_mat, T_mat, color=sol.u[3], colorrange=(1, 1.1), colormap=:matter)
    ax = Axis(fig[1, 3], width=600, height=600)
    mesh!(ax, pt_mat, T_mat, color=sol.u[5], colorrange=(1, 1.1), colormap=:matter)
    SAVE_FIGURE && save("figures/reaction_diffusion_equation_test.png", fig)

    ## Step 6: Define the exact solution for comparison later 
    function reaction_diffusion_exact_solution(x, y, t)
        u_exact = zeros(length(x))
        for i in eachindex(x)
            u_exact[i] = exp(t) * sqrt(besseli(0.0, sqrt(2) * sqrt(x[i]^2 + y[i]^2)))
        end
        return u_exact
    end

    ## Step 7: Compare the results
    all_errs = [Float64[] for _ in eachindex(sol)]
    u_exact = [reaction_diffusion_exact_solution(points[1, :], points[2, :], τ) for τ in sol.t]
    u_fvm = reduce(hcat, sol.u)
    u_exact = reduce(hcat, u_exact)
    errs = reduce(hcat, [100abs.(u - û) / maximum(abs.(u)) for (u, û) in zip(eachcol(u_exact), eachcol(u_fvm))])
    @test all(<(0.1), mean.(eachcol(errs)))
    @test all(<(0.1), median.(eachcol(errs)))
    @test mean(errs) < 0.05
    @test median(errs) < 0.05

    ## Step 8: Visualise the comparison 
    fig = Figure(fontsize=42, resolution=(3586.6597f0, 1466.396f0))
    ax = Axis(fig[1, 1], width=600, height=600, title=L"(a):$ $ Exact solution, $t = %$(sol.t[1])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_exact[:, 1], colorrange=(1, 1.1), colormap=:matter)
    ax = Axis(fig[1, 2], width=600, height=600, title=L"(b):$ $ Exact solution, $t = %$(sol.t[2])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_exact[:, 2], colorrange=(1, 1.1), colormap=:matter)
    ax = Axis(fig[1, 3], width=600, height=600, title=L"(c):$ $ Exact solution, $t = %$(sol.t[3])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_exact[:, 3], colorrange=(1, 1.1), colormap=:matter)
    ax = Axis(fig[1, 4], width=600, height=600, title=L"(d):$ $ Exact solution, $t = %$(sol.t[4])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_exact[:, 4], colorrange=(1, 1.1), colormap=:matter)
    ax = Axis(fig[1, 5], width=600, height=600, title=L"(e):$ $ Exact solution, $t = %$(sol.t[5])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_exact[:, 5], colorrange=(1, 1.1), colormap=:matter)
    ax = Axis(fig[2, 1], width=600, height=600, title=L"(f):$ $ Numerical solution, $t = %$(sol.t[1])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 1], colorrange=(1, 1.1), colormap=:matter)
    ax = Axis(fig[2, 2], width=600, height=600, title=L"(g):$ $ Numerical solution, $t = %$(sol.t[2])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 2], colorrange=(1, 1.1), colormap=:matter)
    ax = Axis(fig[2, 3], width=600, height=600, title=L"(h):$ $ Numerical solution, $t = %$(sol.t[3])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 3], colorrange=(1, 1.1), colormap=:matter)
    ax = Axis(fig[2, 4], width=600, height=600, title=L"(i):$ $ Numerical solution, $t = %$(sol.t[4])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 4], colorrange=(1, 1.1), colormap=:matter)
    ax = Axis(fig[2, 5], width=600, height=600, title=L"(j):$ $ Numerical solution, $t = %$(sol.t[5])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 5], colorrange=(1, 1.1), colormap=:matter)
    SAVE_FIGURE && save("figures/reaction_heat_equation_test_error.png", fig)
end

###########################################################
##
## Example IV: Porous-Medium equation 
##
###########################################################
@testset "Porous-Medium equation" begin
    ## Step 0: Define all the parameters 
    m = 2
    M = 0.37
    D = 2.53
    final_time = 12.0
    ε = 0.1

    ## Step 1: Define the mesh 
    RmM = 4m / (m - 1) * (M / (4π))^((m - 1) / m)
    L = sqrt(RmM) * (D * final_time)^(1 / (2m))
    n = 500
    x₁ = LinRange(-L, L, n)
    x₂ = LinRange(L, L, n)
    x₃ = LinRange(L, -L, n)
    x₄ = LinRange(-L, -L, n)
    y₁ = LinRange(-L, -L, n)
    y₂ = LinRange(-L, L, n)
    y₃ = LinRange(L, L, n)
    y₄ = LinRange(L, -L, n)
    x = reduce(vcat, [x₁, x₂, x₃, x₄])
    y = reduce(vcat, [y₁, y₂, y₃, y₄])
    xy = [(x, y) for (x, y) in zip(x, y)]
    unique!(xy)
    x = getx.(xy)
    y = gety.(xy)
    r = 0.1
    T, adj, adj2v, DG, points, BN = generate_mesh(x, y, r; gmsh_path=GMSH_PATH)
    mesh = FVMGeometry(T, adj, adj2v, DG, points, BN)

    ## Step 2: Define the boundary conditions 
    bc = ((x, y, t, u::T, p) where {T}) -> zero(T)
    types = :D
    BCs = BoundaryConditions(mesh, bc, types, BN)

    ## Step 3: Define the actual PDE  
    f = (x, y) -> M * 1 / (ε^2 * π) * exp(-1 / (ε^2) * (x^2 + y^2))
    diff_fnc = (x, y, t, u, p) -> p[1] * u^(p[2] - 1)
    diff_parameters = (D, m)
    u₀ = [f(points[:, i]...) for i in axes(points, 2)]
    prob = FVMProblem(mesh, BCs; diffusion_function=diff_fnc,
        diffusion_parameters=diff_parameters, initial_condition=u₀, final_time)

    ## Step 4: Solve
    alg = TRBDF2(linsolve=KLUFactorization())
    sol = solve(prob, alg; saveat=3.0)

    ## Step 5: Visualisation 
    pt_mat = Matrix(points')
    T_mat = [collect(T)[i][j] for i in 1:length(T), j in 1:3]
    fig = Figure(resolution=(2131.8438f0, 684.27f0), fontsize=38)
    ax = Axis(fig[1, 1], width=600, height=600)
    mesh!(ax, pt_mat, T_mat, color=sol.u[1], colorrange=(0.0, 0.05), colormap=:matter)
    ax = Axis(fig[1, 2], width=600, height=600)
    mesh!(ax, pt_mat, T_mat, color=sol.u[3], colorrange=(0.0, 0.05), colormap=:matter)
    ax = Axis(fig[1, 3], width=600, height=600)
    mesh!(ax, pt_mat, T_mat, color=sol.u[5], colorrange=(0.0, 0.05), colormap=:matter)
    SAVE_FIGURE && save("figures/porous_medium_test.png", fig)

    ## Step 6: Define the exact solution for comparison later 
    function porous_medium_exact_solution(x, y, t, m, M, D)
        u_exact = zeros(length(x))
        RmM = 4m / (m - 1) * (M / (4π))^((m - 1) / m)
        for i in eachindex(x)
            if x[i]^2 + y[i]^2 < RmM * (D * t)^(1 / m)
                u_exact[i] = (D * t)^(-1 / m) * ((M / (4π))^((m - 1) / m) - (m - 1) / (4m) * (x[i]^2 + y[i]^2) * (D * t)^(-1 / m))^(1 / (m - 1))
            else
                u_exact[i] = 0.0
            end
        end
        return u_exact
    end

    ## Step 7: Compare the results
    all_errs = [Float64[] for _ in eachindex(sol)]
    u_exact = [porous_medium_exact_solution(points[1, :], points[2, :], τ, m, M, D) for τ in sol.t]
    u_fvm = reduce(hcat, sol.u)
    u_exact = reduce(hcat, u_exact)
    errs = reduce(hcat, [100abs.(u - û) / maximum(abs.(u)) for (u, û) in zip(eachcol(u_exact[:, 2:end]), eachcol(u_fvm[:, 2:end]))])
    @test all(<(0.1), mean.(eachcol(errs)))
    @test all(<(0.1), median.(eachcol(errs)))
    @test mean(errs) < 0.05
    @test median(errs) < 0.05

    ## Step 8: Visualise the comparison 
    fig = Figure(fontsize=42, resolution=(3586.6597f0, 1466.396f0))
    ax = Axis(fig[1, 1], width=600, height=600, title=L"(a):$ $ Exact solution, $t = %$(sol.t[1])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_exact[:, 1], colorrange=(0, 0.05), colormap=:matter)
    ax = Axis(fig[1, 2], width=600, height=600, title=L"(b):$ $ Exact solution, $t = %$(sol.t[2])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_exact[:, 2], colorrange=(0, 0.05), colormap=:matter)
    ax = Axis(fig[1, 3], width=600, height=600, title=L"(c):$ $ Exact solution, $t = %$(sol.t[3])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_exact[:, 3], colorrange=(0, 0.05), colormap=:matter)
    ax = Axis(fig[1, 4], width=600, height=600, title=L"(d):$ $ Exact solution, $t = %$(sol.t[4])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_exact[:, 4], colorrange=(0, 0.05), colormap=:matter)
    ax = Axis(fig[1, 5], width=600, height=600, title=L"(e):$ $ Exact solution, $t = %$(sol.t[5])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_exact[:, 5], colorrange=(0, 0.05), colormap=:matter)
    ax = Axis(fig[2, 1], width=600, height=600, title=L"(f):$ $ Numerical solution, $t = %$(sol.t[1])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 1], colorrange=(0, 0.05), colormap=:matter)
    ax = Axis(fig[2, 2], width=600, height=600, title=L"(g):$ $ Numerical solution, $t = %$(sol.t[2])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 2], colorrange=(0, 0.05), colormap=:matter)
    ax = Axis(fig[2, 3], width=600, height=600, title=L"(h):$ $ Numerical solution, $t = %$(sol.t[3])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 3], colorrange=(0, 0.05), colormap=:matter)
    ax = Axis(fig[2, 4], width=600, height=600, title=L"(i):$ $ Numerical solution, $t = %$(sol.t[4])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 4], colorrange=(0, 0.05), colormap=:matter)
    ax = Axis(fig[2, 5], width=600, height=600, title=L"(j):$ $ Numerical solution, $t = %$(sol.t[5])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 5], colorrange=(0, 0.05), colormap=:matter)
    SAVE_FIGURE && save("figures/porous_medium_test_error.png", fig)
end

###########################################################
##
## Example V: The Porous-Medium equation with a linear source
##
###########################################################
## Step 0: Define all the parameters 
m = 3.4
M = 2.3
D = 0.581
λ = 0.2
final_time = 10.0
ε = 0.1

## Step 1: Define the mesh 
RmM = 4m / (m - 1) * (M / (4π))^((m - 1) / m)
L = sqrt(RmM) * (D / (λ * (m - 1)) * (exp(λ * (m - 1) * final_time) - 1))^(1 / (2m))
n = 500
x₁ = LinRange(-L, L, n)
x₂ = LinRange(L, L, n)
x₃ = LinRange(L, -L, n)
x₄ = LinRange(-L, -L, n)
y₁ = LinRange(-L, -L, n)
y₂ = LinRange(-L, L, n)
y₃ = LinRange(L, L, n)
y₄ = LinRange(L, -L, n)
x = reduce(vcat, [x₁, x₂, x₃, x₄])
y = reduce(vcat, [y₁, y₂, y₃, y₄])
xy = [(x, y) for (x, y) in zip(x, y)]
unique!(xy)
x = getx.(xy)
y = gety.(xy)
r = 0.07
T, adj, adj2v, DG, points, BN = generate_mesh(x, y, r; gmsh_path=GMSH_PATH)
mesh = FVMGeometry(T, adj, adj2v, DG, points, BN)

## Step 2: Define the boundary conditions 
bc = ((x, y, t, u::T, p) where {T}) -> zero(T)
types = :D
BCs = BoundaryConditions(mesh, bc, types, BN)

## Step 3: Define the exact solution for comparison later 
function porous_medium_exact_solution(x, y, t, m, M)
    u_exact = zeros(length(x))
    RmM = 4m / (m - 1) * (M / (4π))^((m - 1) / m)
    for i in eachindex(x)
        if x[i]^2 + y[i]^2 < RmM * t^(1 / m)
            u_exact[i] = t^(-1 / m) * ((M / (4π))^((m - 1) / m) - (m - 1) / (4m) * (x[i]^2 + y[i]^2) * t^(-1 / m))^(1 / (m - 1))
        else
            u_exact[i] = 0.0
        end
    end
    return u_exact
end
function porous_medium_linear_source_exact_solution(x, y, t, m, M, D, λ)
    return exp(λ*t) * porous_medium_exact_solution(x, y, D/(λ*(m-1))*(exp(λ*(m-1)*t)-1), m, M)
end

## Step 4: Define the actual PDE  
f = (x, y) -> M * 1 / (ε^2 * π) * exp(-1 / (ε^2) * (x^2 + y^2))
diff_fnc = (x, y, t, u, p) -> p[1] * abs(u)^(p[2] - 1)
reac_fnc = (x, y, t, u, p) -> p[1] * u
diff_parameters = (D, m)
react_parameter = λ
u₀ = [f(points[:, i]...) for i in axes(points, 2)]
prob = FVMProblem(mesh, BCs; diffusion_function=diff_fnc,
    diffusion_parameters=diff_parameters,
    reaction_function=reac_fnc, reaction_parameters=react_parameter,
    initial_condition=u₀, final_time)

## Step 5: Solve
alg = TRBDF2(linsolve=KLUFactorization())
sol = solve(prob, alg; saveat=2.5)

## Step 6: Visualisation 
pt_mat = Matrix(points')
T_mat = [collect(T)[i][j] for i in 1:length(T), j in 1:3]
fig = Figure(resolution=(2131.8438f0, 684.27f0), fontsize=38)
ax = Axis(fig[1, 1], width=600, height=600)
mesh!(ax, pt_mat, T_mat, color=sol.u[1], colorrange=(0.0, 0.5), colormap=:matter)
ax = Axis(fig[1, 2], width=600, height=600)
mesh!(ax, pt_mat, T_mat, color=sol.u[3], colorrange=(0.0, 0.5), colormap=:matter)
ax = Axis(fig[1, 3], width=600, height=600)
mesh!(ax, pt_mat, T_mat, color=sol.u[5], colorrange=(0.0, 0.5), colormap=:matter)
SAVE_FIGURE && save("figures/porous_medium_linear_source_test.png", fig)

## Step 7: Compare the results
all_errs = [Float64[] for _ in eachindex(sol)]
u_exact = [porous_medium_linear_source_exact_solution(points[1, :], points[2, :], τ, m, M, D, λ) for τ in sol.t]
u_fvm = reduce(hcat, sol.u)
u_exact = reduce(hcat, u_exact)
errs = reduce(hcat, [100abs.(u - û) / maximum(abs.(u)) for (u, û) in zip(eachcol(u_exact[:, 2:end]), eachcol(u_fvm[:, 2:end]))])
@test all(<(0.26), mean.(eachcol(errs)))
@test all(<(0.17), median.(eachcol(errs)))
@test mean(errs) < 0.25
@test median(errs) < 0.05

## Step 8: Visualise the comparison 
fig = Figure(fontsize=42, resolution=(3586.6597f0, 1466.396f0))
ax = Axis(fig[1, 1], width=600, height=600, title=L"(a):$ $ Exact solution, $t = %$(sol.t[1])$", titlealign=:left)
mesh!(ax, pt_mat, T_mat, color=u_exact[:, 1], colorrange=(0, 0.05), colormap=:matter)
ax = Axis(fig[1, 2], width=600, height=600, title=L"(b):$ $ Exact solution, $t = %$(sol.t[2])$", titlealign=:left)
mesh!(ax, pt_mat, T_mat, color=u_exact[:, 2], colorrange=(0, 0.05), colormap=:matter)
ax = Axis(fig[1, 3], width=600, height=600, title=L"(c):$ $ Exact solution, $t = %$(sol.t[3])$", titlealign=:left)
mesh!(ax, pt_mat, T_mat, color=u_exact[:, 3], colorrange=(0, 0.05), colormap=:matter)
ax = Axis(fig[1, 4], width=600, height=600, title=L"(d):$ $ Exact solution, $t = %$(sol.t[4])$", titlealign=:left)
mesh!(ax, pt_mat, T_mat, color=u_exact[:, 4], colorrange=(0, 0.05), colormap=:matter)
ax = Axis(fig[1, 5], width=600, height=600, title=L"(e):$ $ Exact solution, $t = %$(sol.t[5])$", titlealign=:left)
mesh!(ax, pt_mat, T_mat, color=u_exact[:, 5], colorrange=(0, 0.05), colormap=:matter)
ax = Axis(fig[2, 1], width=600, height=600, title=L"(f):$ $ Numerical solution, $t = %$(sol.t[1])$", titlealign=:left)
mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 1], colorrange=(0, 0.05), colormap=:matter)
ax = Axis(fig[2, 2], width=600, height=600, title=L"(g):$ $ Numerical solution, $t = %$(sol.t[2])$", titlealign=:left)
mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 2], colorrange=(0, 0.05), colormap=:matter)
ax = Axis(fig[2, 3], width=600, height=600, title=L"(h):$ $ Numerical solution, $t = %$(sol.t[3])$", titlealign=:left)
mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 3], colorrange=(0, 0.05), colormap=:matter)
ax = Axis(fig[2, 4], width=600, height=600, title=L"(i):$ $ Numerical solution, $t = %$(sol.t[4])$", titlealign=:left)
mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 4], colorrange=(0, 0.05), colormap=:matter)
ax = Axis(fig[2, 5], width=600, height=600, title=L"(j):$ $ Numerical solution, $t = %$(sol.t[5])$", titlealign=:left)
mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 5], colorrange=(0, 0.05), colormap=:matter)
SAVE_FIGURE && save("figures/porous_medium_linear_source_test_error.png", fig)
