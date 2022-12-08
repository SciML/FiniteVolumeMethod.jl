"""
    BoundaryEdgeMatrix

Information for the boundary edges. Boundaries are to be interpreted as being counter-clockwise, 
so that an edge has a left and a right node.

# Fields 
- `left_nodes::Vector{Int64}`: The left nodes for each edge, given as indices for the points.
- `right_nodes::Vector{Int64}`: The right nodes for each edge, given as indices for the points. 
- `adjacent_nodes::Vector{Int64}`: The adjacent node to each edge, i.e. the node such that `(left_nodes[i], right_nodes[i], adjacent_nodes[i])` is a positively oriented triangle in the underlying triangular mesh.
- `types::Vector{Int64}`: The boundaries are split into segments, so `types[i]` says what segment number the `i`th edge is on.
"""
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

"""
    OutwardNormalBoundary{N}

Information for the unit outward normal vector field on the boundary. 

# Fields 
- `x_normals::N`: The x-coordinates for the tail of each normal vector, with `x_normals[i]` the coordinate for the `i`th edge.
- `y_normals::N`: The y-coordinates for the tail of each normal vector, with `x_normals[i]` the coordinate for the `i`th edge. 
- `types::Vector{Int64}`: As in [`BoundaryEdgeMatrix`](@ref), `types[i]` says what segment number the `i`th edge is on.
"""
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

"""
    BoundaryInformation{N,BE}

Information for the boundary of the domain.

# Fields 
- `edge_information::BoundaryEdgeMatrix`: The [`BoundaryEdgeMatrix`](@ref).
- `normal_information::OutwardNormalBoundary{N}`: The [`OutwardNormalBoundary`](@ref).
- `boundary_nodes::Vector{Int64}`: The indices for the nodes on the boundary, given in a counter-clockwise order.
- `boundary_elements::BE`: The elements of the underlying triangular mesh that share at least one edge with the boundary.
"""
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

"""
    InteriorInformation{EL,IEBEI}

Information for the interior of the domain. 

# Fields 
- `nodes::Vector{Int64}`: Indices for the all the nodes that are in the interior of the domain. 
- `elements::EL`: The elements of the underlying triangular mesh that share no edges with the boundary. 
- `interior_edge_boundary_element_identifier::IEBEI`: Given a boundary element `V` (see [`BoundaryInformation`](@ref)), this maps `V` to a list of tuples of the form `((v₁, j₁), (v₂, j₂))`, where `(v₁, v₂)` is an edge such that `v₁` is a boundary node and `v₂` is an interior node, and `j₁` and `j₂` give the positions of `v₁` and `v₂`, respectively, in `V`, e.g. `V[j₁] == v₁`. See also [`construct_interior_edge_boundary_element_identifier`](@ref).
"""
struct InteriorInformation{EL,IEBEI}
    nodes::Vector{Int64}
    elements::EL
    interior_edge_boundary_element_identifier::IEBEI
end
get_interior_nodes(II::InteriorInformation) = II.nodes
get_interior_elements(II::InteriorInformation) = II.elements
get_interior_edges(II::InteriorInformation, T) = II.interior_edge_boundary_element_identifier[T]

""" 
    MeshInformation{EL,P,Adj,Adj2V,NGH,TAR}

Information about the underlying triangular mesh. 

# Fields 
- `elements::EL`: All elements of the mesh. 
- `points::P`: The coordinates for the nodes of the mesh. 
- `adjacent::Adj`: The adjacent map for the triangulation. 
- `adjacent2vertex::Adj2V`: The adjacent-to-vertex map for the triangulation. 
- `neighbours::NGH`: The neighbour map for the triangulation, mapping mesh nodes to all other nodes that share an edge with it. 
- `total_area::TAR`: The total area of the mesh.
"""
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

"""
    ElementInformation{CoordinateType,VectorStorage,ScalarStorage,CoefficientStorage,FloatType}

Information for an individual element.

# Fields 
- `centroid::CoordinateType`: The centroid of the element. 
- `midpoints::VectorStorage`: The coordinates for the midpoints of each edge. 
- `control_volume_edge_midpoints::VectorStorage`: For each edge, this is the midpoint of the line that connects the edge to the centroid. 
- `lengths::ScalarStorage`: The lengths for each edge. 
- `shape_function_coefficients::CoefficientStorage`: These are the `sᵢⱼ` coefficients for the element; see their definition in the README.
- `normals::VectorStorage`: The outward unit normals to each edge.
- `area::FloatType`: The area of the element.
"""
struct ElementInformation{CoordinateType,VectorStorage,ScalarStorage,CoefficientStorage,FloatType}
    centroid::CoordinateType
    midpoints::VectorStorage
    control_volume_edge_midpoints::VectorStorage
    lengths::ScalarStorage # are lengths even needed? It just scales the normals when we multiply later, but this just means we undo the normalisation (i.e. we do n̂ * ℓ = (n/ℓ)ℓ = n later)
    shape_function_coefficients::CoefficientStorage
    normals::VectorStorage
    area::FloatType
end
get_centroid(EI::ElementInformation) = EI.centroid
get_midpoints(EI::ElementInformation) = EI.midpoints
get_midpoints(EI::ElementInformation, i) = get_midpoints(EI)[i]
get_control_volume_edge_midpoints(EI::ElementInformation) = EI.control_volume_edge_midpoints
get_control_volume_edge_midpoints(EI::ElementInformation, i) = get_control_volume_edge_midpoints(EI)[i]
get_lengths(EI::ElementInformation) = EI.lengths
get_lengths(EI::ElementInformation, i) = get_lengths(EI)[i]
gets(EI::ElementInformation) = EI.shape_function_coefficients
gets(EI::ElementInformation, i) = gets(EI)[i]
get_normals(EI::ElementInformation) = EI.normals
get_normals(EI::ElementInformation, i) = get_normals(EI)[i]
get_area(EI::ElementInformation) = EI.area

"""
    FVMGeometry{EL,P,Adj,Adj2V,NGH,TAR,N,BE,IEBEI,T,CT,VS,SS,CS,FT}

Geometric information about the underlying triangular mesh, as needed for the finite volume method. 

# Fields 
- `mesh_information::MeshInformation`: Information about the underlying triangular mesh; see [`MeshInformation`](@ref).
- `boundary_information::BoundaryInformation`: Information about the boundary of the underlying triangular mesh; see [`BoundaryInformation`](@ref).
- `interior_information::InteriorInformation`: Information about the interior of the underlying triangular mesh; see [`InteriorInformation`](@ref).
- `element_information_list::Dict{T, ElementInformation}`: List that maps each element to the information about that mesh; see [`ElementInformation`](@ref).
- `volumes::Dict{Int64,FT}`: Maps node indices to the volume of its corresponding control volume.
"""
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
get_control_volume_edge_midpoints(geo::FVMGeometry, T) = get_control_volume_edge_midpoints(get_element_information(geo, T))
get_control_volume_edge_midpoints(geo::FVMGeometry, T, i) = get_control_volume_edge_midpoints(get_element_information(geo, T), i)
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

"""
    FVMGeometry(T::Ts, adj, adj2v, DG, pts, BNV;
        coordinate_type=Vector{number_type(pts)},
        control_volume_storage_type_vector=NTuple{3,coordinate_type},
        control_volume_storage_type_scalar=NTuple{3,number_type(pts)},
        shape_function_coefficient_storage_type=NTuple{9,number_type(pts)},
        interior_edge_storage_type=NTuple{2,Int64},
        interior_edge_pair_storage_type=NTuple{2,interior_edge_storage_type}) where {Ts}

Constructor for [`FVMGeometry`](@ref).

# Arguments 
- `T::Ts`: The list of triangles. 
- `adj`: The adjacent map; see [`DelaunayTriangulation.Adjacent`](@ref).
- `adj2v`: The adjacent-to-vertex map; see [`DelaunayTriangulation.Adjacent2Vertex`](@ref).
- `DG`: The graph for the mesh connectivity; see [`DelaunayTriangulation.DelaunayGraph`](@ref).
- `pts`: The points for the mesh. 
- `BNV`: The boundary node vector. This should be a vector of vectors, where each nested vector is a list of indices that define the nodes for the corresponding segment, and `first(BNV[i]) == last(BNV[i-1])`. The nodes must be listed in counter-clockwise order.

# Keyword Arguments 
- `coordinate_type=Vector{number_type(pts)}`: How coordinates are represented.
- `control_volume_storage_type_vector=NTuple{3,coordinate_type}`: How information for triples of coordinates is represented. The element type must be `coordinate_type`.
- `control_volume_storage_type_scalar=NTuple{3,number_type(pts)}`: How information triples of scalars is represented. The element type must be the same as the element type used for `pts`.
- `shape_function_coefficient_storage_type=NTuple{9,number_type(pts)}`: How the nine shape function coefficients for each element are stored. The element type must be the same as the element type used for `pts`.
- `interior_edge_storage_type=NTuple{2,Int64}`: How the tuples `(v, j)`, as defined for the interior edge identifier in [`InteriorInformation`](@ref), are stored.
- `interior_edge_pair_storage_type=NTuple{2,interior_edge_storage_type}`: How the list of tuples for the interior edge tuples, as defined for the interior edge identifier in [`InteriorInformation`](@ref), are stored. The element type must be `interior_edge_pair_storage_type`.

# Outputs 
The returned value is the [`FVMGeometry`](@ref) object storing information about the underlying triangular mesh.
"""
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
        control_volume_midpoints = compute_control_volume_edge_midpoints(centroid, midpoints; coordinate_type, control_volume_storage_type_vector)
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
        element_infos[τ] = ElementInformation(centroid, midpoints, control_volume_midpoints, lengths, s, normals, element_area)
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

function compute_control_volume_edge_midpoints(m::A, c::A; coordinate_type) where {A} # use ::A to avoid methods being overwritten
    xσ = (getx(m) + getx(c)) / 2
    yσ = (gety(m) + gety(c)) / 2
    σ = construct_coordinate(coordinate_type, xσ, yσ)
    return σ
end
function compute_control_volume_edge_midpoints(centroid, midpoints; coordinate_type, control_volume_storage_type_vector)
    σ₁ = compute_control_volume_edge_midpoints(midpoints[1], centroid; coordinate_type)
    σ₂ = compute_control_volume_edge_midpoints(midpoints[2], centroid; coordinate_type)
    σ₃ = compute_control_volume_edge_midpoints(midpoints[3], centroid; coordinate_type)
    edge_midpoints = construct_control_volume_storage(control_volume_storage_type_vector, σ₁, σ₂, σ₃)
    return edge_midpoints
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
interior edges, represented as `((v₁, j₁), (v₂, j₂))` (a vector of), with `v[j₁] = v₁ → v[j₂] = v₂` an edge of the element that 
is in the interior.
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