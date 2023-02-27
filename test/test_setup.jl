using DelaunayTriangulation
using StaticArrays
using InteractiveUtils
const DT = DelaunayTriangulation
const FVM = FiniteVolumeMethod
const SHOW_WARNTYPE = false
const SAVE_FIGURE = true
const GMSH_PATH = "./gmsh-4.9.4-Windows64/gmsh.exe"
function example_triangulation()
    a, b, c, d = 0.0, 2.0, 0.0, 5.0
    nx = 5
    ny = 6
    tri = triangulate_rectangle(a, b, c, d, nx, ny; single_boundary=false, add_ghost_triangles=false)
    return a, b, c, d, nx, ny, tri
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