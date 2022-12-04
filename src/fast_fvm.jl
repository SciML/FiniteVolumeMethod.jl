##############################################################################
##
## SETUP
##
##############################################################################
using DelaunayTriangulation
using FunctionWrappersWrappers
using PreallocationTools
using StaticArraysCore
using LinearAlgebra
using SparseArrays
using SciMLBase
using DiffEqBase
import DiffEqBase: dualgen

export FVMGeometry
export BoundaryConditions
export FVMProblem
export eval_interpolant
export eval_interpolant!

##############################################################################
##
## FVMGeometry
##
##############################################################################
struct BoundaryEdgeMatrix
    left_nodes::Vector{Int64}
    right_nodes::Vector{Int64}
    adjacent_nodes::Vector{Int64}
    types::Vector{Int64}
end
Base.length(BEM::BoundaryEdgeMatrix) = length(BEM.left_nodes)
Base.eachindex(BEM::BoundaryEdgeMatrix) = eachindex(BEM.left_nodes)
Base.firstindex(BEM::BoundaryEdgeMatrix) = firstindex(BEM.left_nodes)
Base.lastindex(BEM::BoundaryEdgeMatrix) = lastindex(BEM.left_nodes)
get_left_nodes(BEM::BoundaryEdgeMatrix) = BEM.left_nodes
get_boundary_nodes(BEM::BoundaryEdgeMatrix) = get_left_nodes(BEM::BoundaryEdgeMatrix)
get_right_nodes(BEM::BoundaryEdgeMatrix) = BEM.right_nodes
get_adjacent_nodes(BEM::BoundaryEdgeMatrix) = BEM.adjacent_nodes
get_types(BEM::BoundaryEdgeMatrix) = BEM.types
get_left_nodes(BEM::BoundaryEdgeMatrix, i) = get_left_nodes(BEM)[i]
get_right_nodes(BEM::BoundaryEdgeMatrix, i) = get_right_nodes(BEM)[i]
get_adjacent_nodes(BEM::BoundaryEdgeMatrix, i) = get_adjacent_nodes(BEM)[i]
get_types(BEM::BoundaryEdgeMatrix, i) = get_types(BEM)[i]
struct OutwardNormalBoundary{N}
    x_normals::N
    y_normals::N
    types::Vector{Int64}
end
get_x_normals(ONB::OutwardNormalBoundary) = ONB.x_normals
get_y_normals(ONB::OutwardNormalBoundary) = ONB.y_normals
get_types(ONB::OutwardNormalBoundary) = ONB.types
get_x_normals(ONB::OutwardNormalBoundary, i) = get_x_normals(ONB)[i]
get_y_normals(ONB::OutwardNormalBoundary, i) = get_y_normals(ONB)[i]
get_types(ONB::OutwardNormalBoundary, i) = get_types(ONB)[i]
struct BoundaryInformation{N,BE}
    edge_information::BoundaryEdgeMatrix
    normal_information::OutwardNormalBoundary{N}
    boundary_nodes::Vector{Int64}
    boundary_elements::BE
end
get_boundary_nodes(BI::BoundaryInformation) = BI.boundary_nodes
get_boundary_elements(BI::BoundaryInformation) = BI.boundary_elements
get_edge_information(BI::BoundaryInformation) = BI.edge_information
num_boundary_edges(BI::BoundaryInformation) = length(get_edge_information(BI))
struct InteriorInformation{EL,IEBEI}
    nodes::Vector{Int64}
    elements::EL
    interior_edge_boundary_element_identifier::IEBEI
end
get_interior_nodes(II::InteriorInformation) = II.nodes
get_interior_elements(II::InteriorInformation) = II.elements
get_interior_edges(II::InteriorInformation, T) = II.interior_edge_boundary_element_identifier[T]
struct MeshInformation{EL,P,Adj,Adj2V,NGH,TAR}
    elements::EL
    points::P
    adjacent::Adj
    adjacent2vertex::Adj2V
    neighbours::NGH
    total_area::TAR
end
get_neighbours(MI::MeshInformation) = MI.neighbours
DelaunayTriangulation.get_point(MI::MeshInformation, j) = get_point(MI.points, j)
get_points(MI::MeshInformation) = MI.points
get_adjacent(MI::MeshInformation) = MI.adjacent
get_adjacent2vertex(MI::MeshInformation) = MI.adjacent2vertex
get_elements(MI::MeshInformation) = MI.elements
get_element_type(::MeshInformation{EL,P,Adj,Adj2V,NGH,TAR}) where {EL,P,Adj,Adj2V,NGH,TAR} = DelaunayTriangulation.triangle_type(EL)
struct ElementInformation{CoordinateType,VectorStorage,ScalarStorage,CoefficientStorage,FloatType}
    centroid::CoordinateType
    midpoints::VectorStorage
    lengths::ScalarStorage
    shape_function_coefficients::CoefficientStorage
    normals::VectorStorage
    area::FloatType
end
get_centroid(EI::ElementInformation) = EI.centroid
get_midpoints(EI::ElementInformation) = EI.midpoints
get_midpoints(EI::ElementInformation, i) = get_midpoints(EI)[i]
get_lengths(EI::ElementInformation) = EI.lengths
get_lengths(EI::ElementInformation, i) = get_lengths(EI)[i]
gets(EI::ElementInformation) = EI.shape_function_coefficients
gets(EI::ElementInformation, i) = gets(EI)[i]
get_normals(EI::ElementInformation) = EI.normals
get_normals(EI::ElementInformation, i) = get_normals(EI)[i]
get_area(EI::ElementInformation) = EI.area
Base.@kwdef struct FVMGeometry{EL,P,Adj,Adj2V,NGH,TAR,N,BE,IEBEI,T,CT,VS,SS,CS,FT}
    mesh_information::MeshInformation{EL,P,Adj,Adj2V,NGH,TAR}
    boundary_information::BoundaryInformation{N,BE}
    interior_information::InteriorInformation{EL,IEBEI}
    element_information_list::Dict{T,ElementInformation{CT,VS,SS,CS,FT}}
    volumes::Dict{Int64,FT}
end
get_mesh_information(geo::FVMGeometry) = geo.mesh_information
get_boundary_information(geo::FVMGeometry) = geo.boundary_information
get_interior_information(geo::FVMGeometry) = geo.interior_information
get_element_information(geo::FVMGeometry) = geo.element_information_list
get_element_information(geo::FVMGeometry, T) = get_element_information(geo)[T]
get_volumes(geo::FVMGeometry) = geo.volumes
get_volumes(geo::FVMGeometry, i) = get_volumes(geo)[i]
get_boundary_edge_information(geo::FVMGeometry) = get_edge_information(get_boundary_information(geo))
get_interior_nodes(geo::FVMGeometry) = get_interior_nodes(geo.interior_information)
get_boundary_nodes(geo::FVMGeometry) = get_boundary_nodes(geo.boundary_information)
num_boundary_edges(geo::FVMGeometry) = num_boundary_edges(get_boundary_information(geo))
gets(geo::FVMGeometry, T) = gets(get_element_information(geo, T))
gets(geo::FVMGeometry, T, i) = gets(get_element_information(geo, T), i)
get_midpoints(geo::FVMGeometry, T) = get_midpoints(get_element_information(geo, T))
get_midpoints(geo::FVMGeometry, T, i) = get_midpoints(get_element_information(geo, T), i)
get_normals(geo::FVMGeometry, T) = get_normals(get_element_information(geo, T))
get_normals(geo::FVMGeometry, T, i) = get_normals(get_element_information(geo, T), i)
get_lengths(geo::FVMGeometry, T) = get_lengths(get_element_information(geo, T))
get_lengths(geo::FVMGeometry, T, i) = get_lengths(get_element_information(geo, T), i)
get_interior_elements(geo::FVMGeometry) = get_interior_elements(get_interior_information(geo))
get_interior_edges(geo::FVMGeometry, T) = get_interior_edges(get_interior_information(geo), T)
get_boundary_elements(geo::FVMGeometry) = get_boundary_elements(get_boundary_information(geo))
get_neighbours(geo::FVMGeometry) = get_neighbours(get_mesh_information(geo))
get_adjacent(geo::FVMGeometry) = get_adjacent(get_mesh_information(geo))
get_adjacent2vertex(geo::FVMGeometry) = get_adjacent2vertex(get_mesh_information(geo))
DelaunayTriangulation.get_point(geo::FVMGeometry, j) = get_point(get_mesh_information(geo), j)
get_points(geo::FVMGeometry) = get_points(get_mesh_information(geo))
get_elements(geo::FVMGeometry) = get_elements(get_mesh_information(geo))
get_element_type(geo::FVMGeometry) = get_element_type(get_mesh_information(geo))

function FVMGeometry(T::Ts, adj, adj2v, DG, pts, BNV;
    coordinate_type=Vector{number_type(pts)},
    control_volume_storage_type_vector=NTuple{3,coordinate_type},
    control_volume_storage_type_scalar=NTuple{3,number_type(pts)},
    shape_function_coefficient_storage_type=NTuple{9,number_type(pts)},
    interior_edge_storage_type=NTuple{2,Int64},
    interior_edge_pair_storage_type=NTuple{2,interior_edge_storage_type}) where {Ts}
    ## Setup
    num_elements = num_triangles(T)
    num_nodes = num_points(pts)
    float_type = number_type(pts)
    element_type = DelaunayTriangulation.triangle_type(Ts)
    ## Construct the arrays 
    element_infos = Dict{element_type,
        ElementInformation{coordinate_type,control_volume_storage_type_vector,
            control_volume_storage_type_scalar,shape_function_coefficient_storage_type,
            float_type}}()
    vols = Dict{Int64,float_type}(DelaunayTriangulation._eachindex(pts) .=> zero(float_type))
    sizehint!(element_infos, num_elements)
    total_area = zero(float_type)
    ## Loop over each triangle 
    for τ in T
        v₁, v₂, v₃ = indices(τ)
        _pts = (get_point(pts, v₁), get_point(pts, v₂), get_point(pts, v₃))
        centroid = compute_centroid(_pts; coordinate_type)
        midpoints = compute_midpoints(_pts; coordinate_type, control_volume_storage_type_vector)
        edges = control_volume_edges(centroid, midpoints)
        lengths = compute_edge_lengths(edges; control_volume_storage_type_scalar)
        normals = compute_edge_normals(edges, lengths; coordinate_type, control_volume_storage_type_vector)
        p = control_volume_node_centroid(centroid, _pts)
        q = control_volume_connect_midpoints(midpoints)
        S = sub_control_volume_areas(p, q)
        vols[v₁] += S[1]
        vols[v₂] += S[2]
        vols[v₃] += S[3]
        element_area = S[1] + S[2] + S[3]
        s = compute_shape_function_coefficients(_pts; shape_function_coefficient_storage_type)
        total_area += element_area
        element_infos[τ] = ElementInformation(centroid, midpoints, lengths, s, normals, element_area)
    end
    boundary_info = boundary_information(T, adj, pts, BNV)
    boundary_nodes = get_boundary_nodes(boundary_info)
    boundary_elements = get_boundary_elements(boundary_info)
    interior_info = interior_information(boundary_nodes, num_nodes, T, boundary_elements, DG, pts; interior_edge_storage_type, interior_edge_pair_storage_type)
    mesh_info = MeshInformation(T, pts, adj, adj2v, DG, total_area)
    return FVMGeometry(;
        mesh_information=mesh_info,
        boundary_information=boundary_info,
        interior_information=interior_info,
        element_information_list=element_infos,
        volumes=vols)
end

function compute_centroid(p₁, p₂, p₃; coordinate_type)
    cx = (getx(p₁) + getx(p₂) + getx(p₃)) / 3
    cy = (gety(p₁) + gety(p₂) + gety(p₃)) / 3
    c = construct_coordinate(coordinate_type, cx, cy)
    return c
end
function compute_centroid(pts; coordinate_type)
    centroid = compute_centroid(pts[1], pts[2], pts[3]; coordinate_type)
    return centroid
end

construct_coordinate(::Type{A}, x, y) where {F,A<:AbstractVector{F}} = A([x, y])
construct_coordinate(::Type{NTuple{2,F}}, x, y) where {F} = NTuple{2,F}((x, y))
construct_control_volume_storage(::Type{A}, m₁, m₂, m₃) where {F,A<:AbstractVector{F}} = A([m₁, m₂, m₃])
construct_control_volume_storage(::Type{NTuple{3,F}}, m₁, m₂, m₃) where {F} = NTuple{3,F}((m₁, m₂, m₃))

function compute_midpoint(pᵢ, pⱼ; coordinate_type)
    mx = (getx(pᵢ) + getx(pⱼ)) / 2
    my = (gety(pᵢ) + gety(pⱼ)) / 2
    m = construct_coordinate(coordinate_type, mx, my)
    return m
end
function compute_midpoints(pts; coordinate_type, control_volume_storage_type_vector)
    m₁ = compute_midpoint(pts[1], pts[2]; coordinate_type)
    m₂ = compute_midpoint(pts[2], pts[3]; coordinate_type)
    m₃ = compute_midpoint(pts[3], pts[1]; coordinate_type)
    midpoints = construct_control_volume_storage(control_volume_storage_type_vector, m₁, m₂, m₃)
    return midpoints
end

function compute_control_volume_edge(centroid, m)
    e = (getx(centroid) - getx(m), gety(centroid) - gety(m))
    return e
end
function control_volume_edges(centroid, midpoints)
    e₁ = compute_control_volume_edge(centroid, midpoints[1])
    e₂ = compute_control_volume_edge(centroid, midpoints[2])
    e₃ = compute_control_volume_edge(centroid, midpoints[3])
    return (e₁, e₂, e₃)
end

function compute_edge_lengths(edges; control_volume_storage_type_scalar)
    ℓ₁ = norm(edges[1])
    ℓ₂ = norm(edges[2])
    ℓ₃ = norm(edges[3])
    lengths = construct_control_volume_storage(control_volume_storage_type_scalar, ℓ₁, ℓ₂, ℓ₃)
    return lengths
end

function compute_edge_normal(e, ℓ; coordinate_type)
    n = construct_coordinate(coordinate_type, gety(e) / ℓ, -getx(e) / ℓ)
    return n
end
function compute_edge_normals(edges, lengths; coordinate_type, control_volume_storage_type_vector)
    normalised_rotated_e₁_90_cw = compute_edge_normal(edges[1], lengths[1]; coordinate_type)
    normalised_rotated_e₂_90_cw = compute_edge_normal(edges[2], lengths[2]; coordinate_type)
    normalised_rotated_e₃_90_cw = compute_edge_normal(edges[3], lengths[3]; coordinate_type)
    normals = construct_control_volume_storage(control_volume_storage_type_vector,
        normalised_rotated_e₁_90_cw,
        normalised_rotated_e₂_90_cw,
        normalised_rotated_e₃_90_cw
    )
    return normals
end

function compute_control_volume_node_centroid(centroid, p)
    c = (getx(centroid) - getx(p), gety(centroid) - gety(p))
    return c
end
function control_volume_node_centroid(centroid, pts)
    p₁ = compute_control_volume_node_centroid(centroid, pts[1])
    p₂ = compute_control_volume_node_centroid(centroid, pts[2])
    p₃ = compute_control_volume_node_centroid(centroid, pts[3])
    return (p₁, p₂, p₃)
end

function compute_control_volume_connect_midpoints(m₁, m₂)
    q = (getx(m₁) - getx(m₂), gety(m₁) - gety(m₂))
    return q
end
function control_volume_connect_midpoints(midpoints)
    q₁ = compute_control_volume_connect_midpoints(midpoints[1], midpoints[3])
    q₂ = compute_control_volume_connect_midpoints(midpoints[2], midpoints[1])
    q₃ = compute_control_volume_connect_midpoints(midpoints[3], midpoints[2])
    return (q₁, q₂, q₃)
end

function compute_sub_control_volume_area(p, q)
    S = 0.5 * abs(getx(p) * gety(q) - gety(p) * getx(q))
    return S
end
function sub_control_volume_areas(p, q)
    S₁ = compute_sub_control_volume_area(p[1], q[1])
    S₂ = compute_sub_control_volume_area(p[2], q[2])
    S₃ = compute_sub_control_volume_area(p[3], q[3])
    return (S₁, S₂, S₃)
end

function construct_shape_function_coefficient_storage(::Type{A}, s₁, s₂, s₃, s₄, s₅, s₆, s₇, s₈, s₉) where {F,A<:AbstractVector{F}}
    return A([s₁, s₂, s₃, s₄, s₅, s₆, s₇, s₈, s₉])
end
function construct_shape_function_coefficient_storage(::Type{NTuple{9,F}}, s₁, s₂, s₃, s₄, s₅, s₆, s₇, s₈, s₉) where {F}
    return NTuple{9,F}((s₁, s₂, s₃, s₄, s₅, s₆, s₇, s₈, s₉))
end

function compute_shape_function_coefficients(x₁, y₁, x₂, y₂, x₃, y₃)
    Δ = x₂ * y₃ - y₂ * x₃ - x₁ * y₃ + x₃ * y₁ + x₁ * y₂ - x₂ * y₁
    s₁ = (y₂ - y₃) / Δ
    s₂ = (y₃ - y₁) / Δ
    s₃ = (y₁ - y₂) / Δ
    s₄ = (x₃ - x₂) / Δ
    s₅ = (x₁ - x₃) / Δ
    s₆ = (x₂ - x₁) / Δ
    s₇ = (x₂ * y₃ - x₃ * y₂) / Δ
    s₈ = (x₃ * y₁ - x₁ * y₃) / Δ
    s₉ = (x₁ * y₂ - x₂ * y₁) / Δ
    return s₁, s₂, s₃, s₄, s₅, s₆, s₇, s₈, s₉
end
function compute_shape_function_coefficients(pts; shape_function_coefficient_storage_type)
    p₁, p₂, p₃ = pts
    x₁, y₁ = getx(p₁), gety(p₁)
    x₂, y₂ = getx(p₂), gety(p₂)
    x₃, y₃ = getx(p₃), gety(p₃)
    s₁, s₂, s₃, s₄, s₅, s₆, s₇, s₈, s₉ = compute_shape_function_coefficients(x₁, y₁, x₂, y₂, x₃, y₃)
    s = construct_shape_function_coefficient_storage(shape_function_coefficient_storage_type, s₁, s₂, s₃, s₄, s₅, s₆, s₇, s₈, s₉)
    return s
end

function boundary_information(T::Ts, adj, pts, BNV) where {Ts}
    boundary_edges = boundary_edge_matrix(adj, BNV)
    boundary_normals = outward_normal_boundary(pts, boundary_edges)
    boundary_nodes = get_left_nodes(boundary_edges)
    V = construct_triangles(Ts)
    Ttype = DelaunayTriangulation.triangle_type(Ts)
    for n in eachindex(boundary_edges)
        i = get_left_nodes(boundary_edges, n)
        j = get_right_nodes(boundary_edges, n)
        k = get_adjacent_nodes(boundary_edges, n)
        τ = construct_triangle(Ttype, i, j, k)
        DelaunayTriangulation.add_triangle!(V, τ)
    end
    boundary_elements = DelaunayTriangulation.remove_duplicate_triangles(V) # also sorts
    F = DelaunayTriangulation.triangle_type(Ts)
    for τ in boundary_elements
        i, j, k = indices(τ)
        if (i, j, k) ∈ T
            continue
        elseif (j, k, i) ∈ T
            DelaunayTriangulation.delete_triangle!(boundary_elements, τ)
            σ = DelaunayTriangulation.construct_triangle(F, j, k, i)
            DelaunayTriangulation.add_triangle!(boundary_elements, σ)
        elseif (k, i, j) ∈ T
            DelaunayTriangulation.delete_triangle!(boundary_elements, τ)
            σ = DelaunayTriangulation.construct_triangle(F, k, i, j)
            DelaunayTriangulation.add_triangle!(boundary_elements, σ)
        end
    end
    return BoundaryInformation(boundary_edges, boundary_normals, boundary_nodes, boundary_elements)
end

function boundary_edge_matrix(adj, BNV)
    E = Matrix{Int64}(undef, 4, length(unique(reduce(vcat, BNV)))) #[left;right;adjacent[left,right];type]
    all_boundary_nodes = Int64[]
    edge_types = Int64[]
    sizehint!(all_boundary_nodes, size(E, 2))
    sizehint!(edge_types, size(E, 2))
    ## Start by extracting the boundary nodes, and enumerate each edge
    if length(BNV) > 1
        for (n, edge) in enumerate(BNV)
            push!(all_boundary_nodes, edge[1:end-1]...)
            push!(edge_types, repeat([n], length(edge[1:end-1]))...)
        end
    else
        push!(all_boundary_nodes, BNV[1]...)
        push!(edge_types, repeat([1], length(BNV[1]))...)
    end
    ## Now process the rows
    E[1, :] .= all_boundary_nodes
    E[2, :] .= [all_boundary_nodes[2:end]..., all_boundary_nodes[1]]
    for n = 1:size(E, 2)
        E[3, n] = get_edge(adj, E[1, n], E[2, n])
    end
    E[4, :] .= edge_types
    return BoundaryEdgeMatrix(E[1, :], E[2, :], E[3, :], E[4, :])
end

function outward_normal_boundary(pts, E)
    F = number_type(pts)
    x_normals = zeros(F, length(E))
    y_normals = zeros(F, length(E))
    normal_types = zeros(Int64, length(E))
    for n in eachindex(E)
        v₁ = get_left_nodes(E, n)
        v₂ = get_right_nodes(E, n)
        p = get_point(pts, v₁)
        q = get_point(pts, v₂)
        rx = getx(q) - getx(p)
        ry = gety(q) - gety(p)
        ℓ = norm((rx, ry))
        x_normals[n] = ry / ℓ
        y_normals[n] = -rx / ℓ
        normal_types[n] = get_types(E, n)
    end
    N = OutwardNormalBoundary(x_normals, y_normals, normal_types)
    return N
end

function interior_information(boundary_nodes, num_nodes, T::Ts, boundary_elements, DG, pts;
    interior_edge_storage_type, interior_edge_pair_storage_type) where {Ts}
    interior_nodes = setdiff(1:num_nodes, boundary_nodes)
    sorted_T = DelaunayTriangulation.sort_triangles(T)
    sorted_boundary_elements = DelaunayTriangulation.sort_triangles(boundary_elements)
    for τ in sorted_boundary_elements
        DelaunayTriangulation.delete_triangle!(sorted_T, τ)
    end
    interior_elements = sorted_T # need to put this back into the original sorting, so that we can use the correct keys 
    V = DelaunayTriangulation.triangle_type(Ts)
    for τ in interior_elements
        i, j, k = indices(τ)
        if (i, j, k) ∈ T
            continue
        elseif (j, k, i) ∈ T
            DelaunayTriangulation.delete_triangle!(interior_elements, τ)
            σ = DelaunayTriangulation.construct_triangle(V, j, k, i)
            DelaunayTriangulation.add_triangle!(interior_elements, σ)
        elseif (k, i, j) ∈ T
            DelaunayTriangulation.delete_triangle!(interior_elements, τ)
            σ = DelaunayTriangulation.construct_triangle(V, k, i, j)
            DelaunayTriangulation.add_triangle!(interior_elements, σ)
        end
    end
    interior_edge_boundary_element_identifier = construct_interior_edge_boundary_element_identifier(boundary_elements, T, DG, pts;
        interior_edge_storage_type, interior_edge_pair_storage_type)
    return InteriorInformation(interior_nodes, interior_elements, interior_edge_boundary_element_identifier)
end

construct_interior_edge_storage(::Type{A}, v, j) where {F,A<:AbstractVector{F}} = [v, j]
construct_interior_edge_pair_storage(::Type{A}, vj₁, vj₂) where {F,A<:AbstractVector{F}} = F[vj₁, vj₂]
construct_interior_edge_storage(::Type{NTuple{2,F}}, v, j) where {F} = NTuple{2,F}((v, j))
construct_interior_edge_pair_storage(::Type{NTuple{2,F}}, vj₁, vj₂) where {F} = NTuple{2,F}((vj₁, vj₂))

"""
    construct_interior_edge_boundary_element_identifier(boundary_elements, interior_nodes, DT)

Creates a `Dict` that maps a set of nodal indices `v` for some boundary element to the the 
interior edges, represented as `(j₁, j₂)` (a vector of), with `v[j₁] → v[j₂]` an edge of the element that 
is in the interior.
# Implementation details 
This used to be implemented as follows:
```julia 
if v[1] ∈ interior_nodes
    j₁, j₂, j₃ = 1, 2, 3
elseif v[2] ∈ interior_nodes
    j₁, j₂, j₃ = 2, 3, 1
else
    j₁, j₂, j₃ = 3, 1, 2
end
```
But this doesnt work if all the nodes are on the boundary. So, we instead had to refactor 
this code and make it return the interior edges. In particular, we use an approach 
that tests whether each edge is in the interior. We do this by extracting the 
convex hull, and then testing if the midpoint of the edge is inside the convex 
hull using [`ComputationalGeometry.point_in_convex_hull`](@ref). This function 
could be a bit faster, but it's not too bad.
"""
function construct_interior_edge_boundary_element_identifier(boundary_elements, ::Ts, DG, pts;
    interior_edge_storage_type, interior_edge_pair_storage_type) where {Ts}
    V = DelaunayTriangulation.triangle_type(Ts)
    interior_edge_boundary_element_identifier = Dict{V,Vector{interior_edge_pair_storage_type}}()
    sizehint!(interior_edge_boundary_element_identifier, length(boundary_elements))
    idx = convex_hull(DG, pts)
    ch_edges = [(idx[i], idx[i == length(idx) ? 1 : i + 1]) for i in eachindex(idx)]
    for τ in boundary_elements
        v = indices(τ)
        res = Vector{interior_edge_pair_storage_type}()
        sizehint!(res, 2) # # can't have 3 edges in the interior, else it's not a boundary edge
        for (vᵢ, vⱼ) in edges(τ)
            if (vᵢ, vⱼ) ∉ ch_edges # # if this line is an edge of the convex hull, obviously it's not an interior edge
                j₁ = findfirst(vᵢ .== v)
                j₂ = findfirst(vⱼ .== v)
                vᵢj₁ = construct_interior_edge_storage(interior_edge_storage_type, vᵢ, j₁)
                vⱼj₂ = construct_interior_edge_storage(interior_edge_storage_type, vⱼ, j₂)
                vᵢj₁vⱼj₂ = construct_interior_edge_pair_storage(interior_edge_pair_storage_type, vᵢj₁, vⱼj₂)
                push!(res, vᵢj₁vⱼj₂)
            end
        end
        interior_edge_boundary_element_identifier[τ] = res
    end
    return interior_edge_boundary_element_identifier
end

##############################################################################
##
## BoundaryConditions 
##
##############################################################################
is_dirichlet_type(type) = type ∈ (:Dirichlet, :D, :dirichlet, "Dirichlet", "D", "dirichlet")
is_neumann_type(type) = type ∈ (:Neumann, :N, :neumann, "Neumann", "N", "neumann")
is_dudt_type(type) = type ∈ (:Dudt, :dudt, "Dudt", "dudt", "du/dt")

struct BoundaryConditions{BNV,F,P,DN,NN,DuN,INN,BMI,MBI,TM}
    boundary_node_vector::BNV
    functions::F
    parameters::P
    dirichlet_nodes::DN
    neumann_nodes::NN
    dudt_nodes::DuN
    interior_or_neumann_nodes::INN
    boundary_to_mesh_idx::BMI
    mesh_to_boundary_idx::MBI
    type_map::TM
end
function BoundaryConditions(mesh::FVMGeometry, functions, types, boundary_node_vector;
    params=Tuple(nothing for _ in (functions isa Function ? [1] : eachindex(functions))),
    u_type=Float64, float_type=Float64)
    if !(types isa AbstractVector || types isa Tuple)
        types = (types,)
    end
    classified_functions = wrap_functions(functions, params; u_type, float_type)
    dirichlet_nodes, neumann_nodes, dudt_nodes = classify_edges(get_boundary_edge_information(mesh), types)
    interior_or_neumann_nodes = Set{Int64}(union(get_interior_nodes(mesh), neumann_nodes))
    ∂, ∂⁻¹ = boundary_index_maps(get_boundary_edge_information(mesh))
    type_map = construct_type_map(dirichlet_nodes, neumann_nodes, dudt_nodes, ∂⁻¹, get_boundary_edge_information(mesh), types)
    return BoundaryConditions(boundary_node_vector, classified_functions, params,
        dirichlet_nodes, neumann_nodes, dudt_nodes, interior_or_neumann_nodes,
        ∂, ∂⁻¹, type_map
    )
end

@inline get_dirichlet_nodes(BC::BoundaryConditions) = BC.dirichlet_nodes
@inline get_neumann_nodes(BC::BoundaryConditions) = BC.neumann_nodes
@inline get_dudt_nodes(BC::BoundaryConditions) = BC.dudt_nodes
@inline get_interior_or_neumann_nodes(BC::BoundaryConditions) = BC.interior_or_neumann_nodes
@inline get_type_map(BC::BoundaryConditions) = BC.type_map
@inline map_node_to_segment(BC::BoundaryConditions, j) = get_type_map(BC)[j]
@inline is_interior_or_neumann_node(BC::BoundaryConditions, j) = j ∈ get_interior_or_neumann_nodes(BC)
@inline get_boundary_function(BC::BoundaryConditions, idx) = BC.functions[idx]
@inline get_boundary_function_parameters(BC::BoundaryConditions, i) = BC.parameters[i]
@inline evaluate_boundary_function(BC::BoundaryConditions, idx, x, y, t, u) = get_boundary_function(BC, idx)(x, y, t, u, get_boundary_function_parameters(BC, idx))

get_dual_arg_types(::Type{T}, ::Type{U}, ::Type{P}) where {T,U,P} = (
    Tuple{T,T,T,U,P},                       # Typical signature 
    Tuple{T,T,T,dualgen(U),P},              # Signature with "u" a Dual 
    Tuple{T,T,dualgen(T),U,P},              # Signature with "t" a Dual 
    Tuple{T,T,dualgen(T),dualgen(U),P})     # Signature with "u" and "t" Duals
get_dual_ret_types(::Type{U}, ::Type{T}) where {U,T} = (U, dualgen(U), dualgen(T), dualgen(promote_type(U, T)))
function wrap_functions(functions, params; float_type::Type{T}=Float64, u_type::Type{U}=Float64) where {T,U}
    if !(functions isa AbstractVector || functions isa Tuple)
        functions = (functions,)
    end
    if !(params isa AbstractVector || params isa Tuple)
        params = (params,)
    end
    all_arg_types = ntuple(i -> get_dual_arg_types(T, U, typeof(params[i])), length(params))
    all_ret_types = ntuple(i -> get_dual_ret_types(U, T), length(params))
    wrapped_functions = ntuple(i -> FunctionWrappersWrapper(functions[i], all_arg_types[i], all_ret_types[i]), length(params))
    return wrapped_functions
end

function classify_edge_type(current_type, previous_type)
    if is_dirichlet_type(current_type) || is_dirichlet_type(previous_type)
        return :D
    elseif is_dudt_type(current_type) || is_dudt_type(previous_type)
        return :dudt
    elseif is_neumann_type(current_type) && is_neumann_type(previous_type)
        return :N
    else
        throw("Invalid type specification.")
    end
end

function classify_edges(edge_info::BoundaryEdgeMatrix, edge_types)
    num_edges = length(edge_info)
    dirichlet_nodes = Vector{Int64}([])
    neumann_nodes = Vector{Int64}([])
    dudt_nodes = Vector{Int64}([])
    sizehint!(dirichlet_nodes, num_edges)
    sizehint!(neumann_nodes, num_edges)
    sizehint!(dudt_nodes, num_edges)
    for i in eachindex(edge_info)
        edge_node = get_left_nodes(edge_info, i)
        segment_number = get_types(edge_info, i)
        prev_segment_number = get_types(edge_info, i == firstindex(edge_info) ? lastindex(edge_info) : i - 1)
        current_type = edge_types[segment_number]
        prev_type = edge_types[prev_segment_number]
        merged_type = classify_edge_type(current_type, prev_type)
        if is_dirichlet_type(merged_type)
            push!(dirichlet_nodes, edge_node)
        elseif is_neumann_type(merged_type)
            push!(neumann_nodes, edge_node)
        elseif is_dudt_type(merged_type)
            push!(dudt_nodes, edge_node)
        end
    end
    return dirichlet_nodes, neumann_nodes, dudt_nodes
end

function boundary_index_maps(edge_info::BoundaryEdgeMatrix)
    boundary_nodes = get_boundary_nodes(edge_info)
    ∂ = Dict(eachindex(edge_info) .=> boundary_nodes)
    ∂⁻¹ = Dict(boundary_nodes .=> eachindex(edge_info))
    return ∂, ∂⁻¹
end

function construct_type_map(dirichlet_nodes, neumann_nodes, dudt_nodes, ∂⁻¹, edge_info, types)
    type_map = Dict{Int64,Int64}([])
    sizehint!(type_map, length(edge_info))
    for (nodes, is_type) in zip((dirichlet_nodes, neumann_nodes, dudt_nodes), (is_dirichlet_type, is_neumann_type, is_dudt_type))
        for j ∈ nodes
            ∂j = ∂⁻¹[j]
            segment_number = get_types(edge_info, ∂j)
            # Need to check that the edge type matches the function type. If it doesn't, it's because of the previous segment, so we go back 1 segment.
            if !is_type(types[segment_number])
                segment_number = segment_number == 1 ? length(types) : segment_number - 1
            end
            type_map[j] = segment_number
        end
    end
    return type_map
end

##############################################################################
##
## FVMProblem
##
##############################################################################
struct FVMProblem{iip_flux,FG,BC,F,FP,R,RP,IC,FT}
    mesh::FG
    boundary_conditions::BC
    flux_function::F
    flux_parameters::FP
    reaction_function::R
    reaction_parameters::RP
    initial_condition::IC
    initial_time::FT
    final_time::FT
    steady::Bool
end
function FVMProblem(mesh, boundary_conditions;
    iip_flux=true,
    diffusion_function=nothing,
    diffusion_parameters=nothing,
    reaction_function=nothing,
    reaction_parameters=nothing,
    delay_function=nothing,
    delay_parameters=nothing,
    flux_function=nothing,
    flux_parameters=nothing,
    initial_condition,
    initial_time=0.0,
    final_time,
    steady=false,
    q_storage=SVector{2,Float64})
    updated_flux_fnc = construct_flux_function(iip_flux, flux_function,
        delay_function, delay_parameters,
        diffusion_function, diffusion_parameters;
        q_storage)
    updated_reaction_fnc = construct_reaction_function(reaction_function, reaction_parameters,
        delay_function, delay_parameters)
    return FVMProblem{iip_flux,
        typeof(mesh),typeof(boundary_conditions),
        typeof(updated_flux_fnc),typeof(flux_parameters),
        typeof(updated_reaction_fnc),typeof(reaction_parameters),
        typeof(initial_condition),
        typeof(initial_time)}(
        mesh, boundary_conditions,
        updated_flux_fnc, flux_parameters,
        updated_reaction_fnc, reaction_parameters,
        initial_condition,
        initial_time, final_time,
        steady
    )
end
SciMLBase.isinplace(::FVMProblem{iip_flux,FG,BC,F,FP,R,RP,IC,FT}) where {iip_flux,FG,BC,F,FP,R,RP,IC,FT} = iip_flux
get_boundary_conditions(prob::FVMProblem) = prob.boundary_conditions
get_initial_condition(prob::FVMProblem) = prob.initial_condition
get_initial_time(prob::FVMProblem) = prob.initial_time
get_final_time(prob::FVMProblem) = prob.final_time
get_time_span(prob::FVMProblem) = (get_initial_time(prob), get_final_time(prob))

store_q(::Type{A}, x, y) where {A} = A((x, y))
store_q(::Type{Vector{F}}, x, y) where {F} = F[x, y]
store_q(::Type{Vector{F}}, x::H, y::H) where {F,H<:DiffEqBase.ForwardDiff.Dual} = H[x, y]

function construct_flux_function(iip_flux,
    flux_function,
    delay_function, delay_parameters,
    diffusion_function, diffusion_parameters; q_storage::Type{A}=SVector{2,Float64}) where {A}
    if isnothing(flux_function)
        if !isnothing(delay_function)
            if iip_flux
                flux_fnc = (q, x, y, t, α, β, γ, p) -> begin
                    u = α * x + β * y + γ
                    q[1] = -delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * α
                    q[2] = -delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * β
                    return nothing
                end
                return flux_fnc
            else
                flux_fnc = (x, y, t, α, β, γ, p) -> begin
                    u = α * x + β * y + γ
                    q1 = -delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * α
                    q2 = -delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * β
                    return store_q(A, q1, q2)
                end
                return flux_fnc
            end
            return flux_fnc
        else
            if iip_flux
                flux_fnc = (q, x, y, t, α, β, γ, p) -> begin
                    u = α * x + β * y + γ
                    q[1] = -diffusion_function(x, y, t, u, diffusion_parameters) * α
                    q[2] = -diffusion_function(x, y, t, u, diffusion_parameters) * β
                    return nothing
                end
                return flux_fnc
            else
                flux_fnc = (x, y, t, α, β, γ, p) -> begin
                    u = α * x + β * y + γ
                    q1 = -diffusion_function(x, y, t, u, diffusion_parameters) * α
                    q2 = -diffusion_function(x, y, t, u, diffusion_parameters) * β
                    return store_q(A, q1, q2)
                end
                return flux_fnc
            end
            return flux_fnc
        end
    else
        return flux_function
    end
end

function construct_reaction_function(reaction_function, reaction_parameters,
    delay_function, delay_parameters)
    if !isnothing(reaction_function)
        if !isnothing(delay_function)
            reaction_fnc = (x, y, t, u, p) -> begin
                return delay_function(x, y, t, u, delay_parameters) * reaction_function(x, y, t, u, reaction_parameters)
            end
            return reaction_fnc
        else
            return reaction_function
        end
    else
        reaction_fnc = ((x, y, t, u::T, p) where {T}) -> zero(T)
        return reaction_fnc
    end
end

@inline get_mesh(prob::FVMProblem) = prob.mesh
@inline gets(prob::FVMProblem, T) = gets(get_mesh(prob), T)
@inline gets(prob::FVMProblem, T, i) = gets(prob, T)[i]
@inline get_midpoints(prob::FVMProblem, T) = get_midpoints(get_mesh(prob), T)
@inline get_midpoints(prob::FVMProblem, T, i) = get_midpoints(prob, T)[i]
@inline get_normals(prob::FVMProblem, T) = get_normals(get_mesh(prob), T)
@inline get_normals(prob::FVMProblem, T, i) = get_normals(prob, T)[i]
@inline get_lengths(prob::FVMProblem, T) = get_lengths(get_mesh(prob), T)
@inline get_lengths(prob::FVMProblem, T, i) = get_lengths(prob, T)[i]
@inline get_flux(prob::FVMProblem, x, y, t, α, β, γ) = prob.flux_function(x, y, t, α, β, γ, prob.flux_parameters)
@inline get_flux!(flux_cache, prob::FVMProblem, x, y, t, α, β, γ) = prob.flux_function(flux_cache, x, y, t, α, β, γ, prob.flux_parameters)
@inline get_interior_or_neumann_nodes(prob::FVMProblem) = get_interior_or_neumann_nodes(get_boundary_conditions(prob))
@inline get_interior_elements(prob::FVMProblem) = get_interior_elements(get_mesh(prob))
@inline get_interior_edges(prob::FVMProblem, T) = get_interior_edges(get_mesh(prob), T)
@inline get_boundary_elements(prob::FVMProblem) = get_boundary_elements(get_mesh(prob))
@inline DelaunayTriangulation.get_point(prob::FVMProblem, j) = get_point(get_mesh(prob), j)
@inline get_points(prob::FVMProblem) = get_points(get_mesh(prob))
@inline get_volumes(prob::FVMProblem, j) = get_volumes(get_mesh(prob), j)
@inline get_reaction(prob::FVMProblem, x, y, t, u) = prob.reaction_function(x, y, t, u, prob.reaction_parameters)
@inline get_dirichlet_nodes(prob::FVMProblem) = get_dirichlet_nodes(get_boundary_conditions(prob))
@inline get_neumann_nodes(prob::FVMProblem) = get_neumann_nodes(get_boundary_conditions(prob))
@inline get_dudt_nodes(prob::FVMProblem) = get_dudt_nodes(get_boundary_conditions(prob))
@inline get_boundary_nodes(prob::FVMProblem) = get_boundary_nodes(get_mesh(prob))
@inline get_boundary_function_parameters(prob::FVMProblem, i) = get_boundary_function_parameters(get_boundary_conditions(prob), i)
@inline get_neighbours(prob::FVMProblem) = get_neighbours(get_mesh(prob))
@inline map_node_to_segment(prob::FVMProblem, j) = map_node_to_segment(get_boundary_conditions(prob), j)
@inline is_interior_or_neumann_node(prob::FVMProblem, j) = is_interior_or_neumann_node(get_boundary_conditions(prob), j)
@inline evaluate_boundary_function(prob::FVMProblem, idx, x, y, t, u) = evaluate_boundary_function(get_boundary_conditions(prob), idx, x, y, t, u)
@inline DelaunayTriangulation.num_points(prob::FVMProblem) = DelaunayTriangulation.num_points(get_points(prob))
@inline num_boundary_edges(prob::FVMProblem) = num_boundary_edges(get_mesh(prob)) # This will be the same value as the above, but this makes the code clearer to read
@inline get_adjacent(prob::FVMProblem) = get_adjacent(get_mesh(prob))
@inline get_adjacent2vertex(prob::FVMProblem) = get_adjacent2vertex(get_mesh(prob))
@inline get_elements(prob::FVMProblem) = get_elements(get_mesh(prob))
@inline get_element_type(prob::FVMProblem) = get_element_type(get_mesh(prob))

##############################################################################
##
## FVM Equations 
##
##############################################################################
getα(shape_coeffs) = shape_coeffs[1]
getβ(shape_coeffs) = shape_coeffs[2]
getγ(shape_coeffs) = shape_coeffs[3]

@inline function linear_shape_function_coefficients!(shape_coeffs, u, prob, T)
    i, j, k = indices(T)
    shape_coeffs[1] = gets(prob, T, 1) * u[i] + gets(prob, T, 2) * u[j] + gets(prob, T, 3) * u[k]
    shape_coeffs[2] = gets(prob, T, 4) * u[i] + gets(prob, T, 5) * u[j] + gets(prob, T, 6) * u[k]
    shape_coeffs[3] = gets(prob, T, 7) * u[i] + gets(prob, T, 8) * u[j] + gets(prob, T, 9) * u[k]
    return nothing
end

function fvm_eqs_edge!(du, t, (vj, j), (vjnb, jnb), α, β, γ, prob, flux_cache, T)
    x, y = get_midpoints(prob, T, j)
    xn, yn = get_normals(prob, T, j)
    ℓ = get_lengths(prob, T, j)
    if isinplace(prob)
        get_flux!(flux_cache, prob, x, y, t, α, β, γ)
    else
        flux_cache = get_flux(prob, x, y, t, α, β, γ) # Make no assumption that flux_cache is mutable 
    end
    summand = -(getx(flux_cache) * xn + gety(flux_cache) * yn) * ℓ
    if is_interior_or_neumann_node(prob, vj)
        du[vj] += summand
    end
    if is_interior_or_neumann_node(prob, vjnb)
        du[vjnb] -= summand
    end
    return nothing
end
@inline function fvm_eqs_edge!(du, t, α, β, γ, prob, flux_cache, T)
    i, j, k = indices(T)
    fvm_eqs_edge!(du, t, (i, 1), (j, 2), α, β, γ, prob, flux_cache, T)#unrolled
    fvm_eqs_edge!(du, t, (j, 2), (k, 3), α, β, γ, prob, flux_cache, T)
    fvm_eqs_edge!(du, t, (k, 3), (i, 1), α, β, γ, prob, flux_cache, T)
    return nothing
end

@inline function fvm_eqs_interior_element!(du, u, t, prob, shape_coeffs, flux_cache, T)
    linear_shape_function_coefficients!(shape_coeffs, u, prob, T)
    α = getα(shape_coeffs)
    β = getβ(shape_coeffs)
    γ = getγ(shape_coeffs)
    fvm_eqs_edge!(du, t, α, β, γ, prob, flux_cache, T)
    return nothing
end
@inline function fvm_eqs_interior_element!(du, u, t, prob, shape_coeffs, flux_cache)
    for V in get_interior_elements(prob)
        fvm_eqs_interior_element!(du, u, t, prob, shape_coeffs, flux_cache, V)
    end
    return nothing
end

@inline function fvm_eqs_boundary_element!(du, u, t, prob, shape_coeffs, flux_cache, T)
    linear_shape_function_coefficients!(shape_coeffs, u, prob, T)
    α = getα(shape_coeffs)
    β = getβ(shape_coeffs)
    γ = getγ(shape_coeffs)
    interior_edges = get_interior_edges(prob, T)
    for ((vj, j), (vjnb, jnb)) in interior_edges
        fvm_eqs_edge!(du, t, (vj, j), (vjnb, jnb), α, β, γ, prob, flux_cache, T)
    end
    return nothing
end
@inline function fvm_eqs_boundary_element!(du, u, t, prob, shape_coeffs, flux_cache)
    for V in get_boundary_elements(prob)
        fvm_eqs_boundary_element!(du, u, t, prob, shape_coeffs, flux_cache, V)
    end
    return nothing
end

@inline function fvm_eqs_source_contribution!(du, u, t, j, prob)
    x, y = get_point(prob, j)
    V = get_volumes(prob, j)
    R = get_reaction(prob, x, y, t, u[j])
    du[j] = du[j] / V + R
    return nothing
end
@inline function fvm_eqs_source_contribution!(du, u, t, prob)
    for j ∈ get_interior_or_neumann_nodes(prob)
        fvm_eqs_source_contribution!(du, u, t, j, prob)
    end
    return nothing
end

@inline function evaluate_boundary_function(u, t, j, prob)
    segment_number = map_node_to_segment(prob, j)
    x, y = get_point(prob, j)
    val = evaluate_boundary_function(prob, segment_number, x, y, t, u[j])
    return val
end
@inline function evaluate_boundary_function!(du, u, t, j, prob)
    du[j] = evaluate_boundary_function(u, t, j, prob)
    return nothing
end

@inline function update_dudt_nodes!(du, u, t, prob)
    for j in get_dudt_nodes(prob)
        evaluate_boundary_function!(du, u, t, j, prob)
    end
    return nothing
end

function fvm_eqs!(du::AbstractVector{T}, u, p, t) where {T}
    prob, flux_cache, shape_coeffs = p
    tmp_flux_cache = get_tmp(flux_cache, u)
    tmp_shape_coeffs = get_tmp(shape_coeffs, u)
    fill!(du, zero(T))
    fvm_eqs_interior_element!(du, u, t, prob, tmp_shape_coeffs, tmp_flux_cache)
    fvm_eqs_boundary_element!(du, u, t, prob, tmp_shape_coeffs, tmp_flux_cache)
    fvm_eqs_source_contribution!(du, u, t, prob)
    update_dudt_nodes!(du, u, t, prob)
    return nothing
end

function update_dirichlet_nodes!(u, t, prob)
    for j in get_dirichlet_nodes(prob)
        u[j] = evaluate_boundary_function(u, t, j, prob)
    end
    return nothing
end
function update_dirichlet_nodes!(integrator)
    update_dirichlet_nodes!(integrator.u, integrator.t, integrator.p[1])
    return nothing
end

##############################################################################
##
## SOLVER
##
##############################################################################
@inline function dirichlet_callback(no_saveat=false)
    condition = (u, t, integrator) -> true
    cb = DiffEqBase.DiscreteCallback(condition, update_dirichlet_nodes!; save_positions=(no_saveat, no_saveat))
    return cb
end

function jacobian_sparsity(prob::FVMProblem)
    DG = get_neighbours(prob)
    #jac_sparsity = DelaunayTriangulation.adjacency(DelaunayTriangulation.graph(DG)) + I
    # ^ The above creates a dense matrix. We can be a bit smart about this and create the 
    #   coordinate format directly for the sparse matrix conversion.
    has_ghost_edges = DelaunayTriangulation.BoundaryIndex ∈ DelaunayTriangulation.graph(DG).V
    num_nnz = 2(length(edges(DG)) - num_boundary_edges(prob) * has_ghost_edges) + num_points(prob) # Logic: For each edge in the triangulation we obtain two non-zero entries. If each boundary edge is adjoined with a ghost edge, though, then we need to make sure we don't count the contributions from those edges - hence why we subtract it off. Finally, the Jacobian needs to also include the node's relationship with itself, so we add on the number of points.
    I = zeros(Int64, num_nnz)   # row indices 
    J = zeros(Int64, num_nnz)   # col indices 
    V = ones(num_nnz)           # values (all 1)
    ctr = 1
    for i in DelaunayTriangulation._eachindex(get_points(prob))
        I[ctr] = i
        J[ctr] = i
        ctr += 1
        ngh = DelaunayTriangulation.get_neighbour(DG, i)
        for j in ngh
            if has_ghost_edges && j ≠ DelaunayTriangulation.BoundaryIndex
                I[ctr] = i
                J[ctr] = j
                ctr += 1
            end
        end
    end
    return sparse(I, J, V)
end

function SciMLBase.ODEProblem(prob::FVMProblem;
    cache_eltype::Type{F}=eltype(get_initial_condition(prob)),
    jac_prototype=float.(jacobian_sparsity(prob)),
    parallel=false,
    no_saveat=true,
    specialization::Type{S}=SciMLBase.AutoSpecialize,
    chunk_size=PreallocationTools.ForwardDiff.pickchunksize(length(get_initial_condition(prob)))) where {S,F}
    time_span = get_time_span(prob)
    initial_condition = get_initial_condition(prob)
    cb = dirichlet_callback(no_saveat)
    flux_cache = dualcache(zeros(F, 2), chunk_size)
    shape_coeffs = dualcache(zeros(F, 3), chunk_size)
    if parallel
        f = ODEFunction{true,S}(par_fvm_eqs!; jac_prototype)
    else
        f = ODEFunction{true,S}(fvm_eqs!; jac_prototype)
    end
    p = (prob, flux_cache, shape_coeffs)
    ode_problem = ODEProblem{true,S}(f, initial_condition, time_span, p; callback=cb)
    return ode_problem
end

function SciMLBase.solve(prob::FVMProblem, alg;
    cache_eltype::Type{F}=eltype(get_initial_condition(prob)),
    jac_prototype=float.(jacobian_sparsity(prob)),
    parallel=false,
    specialization::Type{S}=SciMLBase.AutoSpecialize,
    chunk_size=PreallocationTools.ForwardDiff.pickchunksize(length(get_initial_condition(prob))),
    kwargs...) where {S,F}
    no_saveat = :saveat ∉ keys(Dict(kwargs))
    ode_problem = ODEProblem(prob; cache_eltype, jac_prototype, parallel, no_saveat, specialization, chunk_size)
    sol = solve(ode_problem, alg; kwargs...)
    return sol
end

##############################################################################
##
## Interpolant 
##
##############################################################################
@inline function eval_interpolant!(αβγ, prob::FVMProblem, x, y, T, u)
    if T ∉ get_elements(prob)
        T = DelaunayTriangulation.shift_triangle_1(T)
    end
    if T ∉ get_elements(prob)
        T = DelaunayTriangulation.shift_triangle_1(T)
    end
    linear_shape_function_coefficients!(αβγ, u, prob, T)
    α = getα(αβγ)
    β = getβ(αβγ)
    γ = getγ(αβγ)
    return α * x + β * y + γ
end
@inline function eval_interpolant(sol, x, y, t_idx::Integer, T)
    prob = sol.prob.p[1]
    shape_coeffs = sol.prob.p[3]
    new_shape_coeffs = get_tmp(shape_coeffs, x)
    return eval_interpolant!(new_shape_coeffs, prob, x, y, T, sol.u[t_idx])
end
@inline function eval_interpolant(sol, x, y, t::Number, T)
    prob = sol.prob.p[1]
    shape_coeffs = sol.prob.p[3]
    new_shape_coeffs = get_tmp(shape_coeffs, x)
    return eval_interpolant!(new_shape_coeffs, prob, x, y, T, sol(t))
end

function DelaunayTriangulation.jump_and_march(x, y, prob::FVMProblem;
    pt_idx=DelaunayTriangulation._eachindex(get_points(prob)),
    m=ceil(Int64, length(pt_idx)^(1 / 3)),
    try_points=(),
    k=DelaunayTriangulation.select_initial_point(get_points(prob), (x, y); m, pt_idx, try_points),
    TriangleType::Type{V}=get_element_type(prob)) where {V}
    adj = get_adjacent(prob)
    adj2v = get_adjacent2vertex(prob)
    DG = get_neighbours(prob)
    pts = get_points(prob)
    q = (x, y)
    return jump_and_march(q, adj, adj2v, DG, pts; pt_idx, m, k, TriangleType)
end