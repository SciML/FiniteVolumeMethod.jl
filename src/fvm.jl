##############################################################################
##
## SETUP
##
##############################################################################
using StatsBase
using LinearAlgebra
using DelaunayTriangulation
using FunctionWranglers
using PreallocationTools
using Unrolled
using FastClosures
using CommonSolve 
using DifferentialEquations
import SimpleGraphs: adjacency
import SparseArrays: sparse
const DIRICHLET_ARGS = 4
const NEUMANN_ARGS = 3
const DUDT_ARGS = 5

##############################################################################
##
## GEOMETRY 
##
##############################################################################
"""
    struct FVMGeometry 

Struct representing the geometry for an [`FVMProblem`](@ref).

# Fields 
- `elements`

Triangular elements in the mesh.
- `interior_elements`

A subset of `elements`, this gives the interior elements of the mesh.
- `boundary_elements`

A subset of `elements`, this gives the boundary elements of the mesh. 
- `points`

The points in the mesh. 
- `interior_points`

Indices of the points in `points` that correspond to interior points. 
- `boundary_points`

Indices of the points in `points` that correspond to boundary points. 
- `adj`

A map that takes an edge `(u, v)` of a triangular element to a vertex `w` such that `(u, v, w) ∈ elements` is positively oriented. 
- `adj2v`

A map that takes a vertex `u` and maps it to the set of edges `(v, w)` such that all the `(u, v, w) ∈ elements` are positively oriented. 
- `neighbours`

A map that takes a vertex `u` and returns the set of all `v` such that `(u, v)` is an edge of some triangular element, i.e. the set of all neighbours to `u`.
- `boundary_node_vector`

Vector of vectors giving the indices for the boundary points, separated by boundary type.
- `boundary_edges`

Edges in the mesh that form the boundary. 
- `boundary_normals`

Normal vectors of the edges in `boundary_edges` that point away from the interior. 
- `interior_edge_boundary_element_identifier`

Given a boundary element, identifies the edges corresponding to interior edges.
- `volumes`

Volumes of the individual control volumes in the mesh.  
- `normals`

Normals of the edges in the individual control volumes in the mesh.
- `shape_function_coeffs`

Coefficients for the linear shape functions in the mesh over each element.
- `areas`

Areas for each triangular element. 
- `total_area`

The total area of the geometry. 
- `weighted_areas`

The values `areas/total_area`.
- `centroids`

The centroid for each triangular element. 
- `lengths`

Length for each control volume edge. 
- `midpoints`

Midpoints for each control volume edge.
"""
Base.@kwdef struct FVMGeometry{
    EL,IE,BEL,P,IP,BP,
    Adj,Adj2V,NGH,
    BNV,BED,BN,IEBEI,
    V,NO,SFC,AR,TAR,WAR,C,L,M
}
    elements::EL
    interior_elements::IE
    boundary_elements::BEL
    points::P
    interior_points::IP
    boundary_points::BP
    adj::Adj
    adj2v::Adj2V
    neighbours::NGH
    boundary_node_vector::BNV
    boundary_edges::BED
    boundary_normals::BN
    interior_edge_boundary_element_identifier::IEBEI
    volumes::V
    normals::NO
    shape_function_coeffs::SFC
    areas::AR
    total_area::TAR
    weighted_areas::WAR
    centroids::C
    lengths::L
    midpoints::M
end
function FVMGeometry(T::Ts, adj, adj2v, DG, pts, BNV) where {Ts}
    num_elements = num_triangles(T)
    num_nodes = num_points(pts)
    F = number_type(pts)
    V = DelaunayTriangulation.triangle_type(Ts)
    midpoints = Dict{V,NTuple{3,Vector{F}}}()
    lengths = Dict{V,NTuple{3,F}}()
    normals = Dict{V,NTuple{3,Vector{F}}}()
    centroids = Dict{V,Vector{F}}()
    s = Dict{V,NTuple{9,F}}()
    vols = Dict{Int64,F}(DelaunayTriangulation._eachindex(pts) .=> zero(F))
    areas = Dict{V,F}()
    weighted_areas = Dict{V,F}()
    sizehint!(midpoints, num_elements)
    sizehint!(lengths, num_elements)
    sizehint!(normals, num_elements)
    sizehint!(centroids, num_elements)
    sizehint!(s, num_elements)
    sizehint!(vols, num_nodes)
    sizehint!(areas, num_elements)
    sizehint!(weighted_areas, num_elements)
    edges = Vector{Vector{F}}(undef, 3)
    p = Vector{Vector{F}}(undef, 3)
    q = Vector{Vector{F}}(undef, 3)
    S = zeros(F, 3)
    total_area = zero(F)
    for τ in T
        v₁, v₂, v₃ = indices(τ)
        _pts = (_get_point(pts, v₁), _get_point(pts, v₂), _get_point(pts, v₃))
        centroid!(centroids, τ, _pts)
        midpoint!(midpoints, τ, _pts)
        control_volume_edges!(edges, centroids, midpoints, τ)
        edge_lengths!(lengths, τ, edges)
        edge_normals!(normals, τ, edges, lengths)
        control_volume_node_centroid!(p, centroids, τ, _pts)
        control_volume_connect_midpoints!(q, midpoints, τ)
        sub_control_volume_areas!(S, p, q)
        vols[v₁] += S[1]
        vols[v₂] += S[2]
        vols[v₃] += S[3]
        areas[τ] = S[1] + S[2] + S[3]
        shape_function_coefficients!(s, τ, _pts)
        total_area += areas[τ]
    end
    for τ in T
        weighted_areas[τ] = areas[τ] / total_area
    end
    boundary_edges, boundary_normals, boundary_nodes, boundary_elements = boundary_information(T, adj, pts, BNV)
    interior_nodes, interior_elements, interior_edge_boundary_element_identifier = interior_information(boundary_nodes, num_nodes, T, boundary_elements, DG, pts)
    return FVMGeometry(;
        elements=T,
        interior_elements,
        boundary_elements,
        points=pts,
        interior_points=interior_nodes,
        boundary_points=boundary_nodes,
        adj,
        adj2v,
        neighbours=DG,
        boundary_node_vector=BNV,
        boundary_edges,
        boundary_normals,
        interior_edge_boundary_element_identifier,
        volumes=vols,
        normals,
        shape_function_coeffs=s,
        areas,
        total_area,
        weighted_areas,
        centroids,
        lengths,
        midpoints)
end

function centroid!(centroids, T, pts)
    centroids[T] = mean(pts)
    return nothing
end

function midpoint!(midpoints, T, pts)
    m₁ = mean((pts[1], pts[2]))
    m₂ = mean((pts[2], pts[3]))
    m₃ = mean((pts[3], pts[1]))
    midpoints[T] = (m₁, m₂, m₃)
    return nothing
end

function control_volume_edges!(edges, centroids, midpoints, T)
    edges[1] = centroids[T] - midpoints[T][1]
    edges[2] = centroids[T] - midpoints[T][2]
    edges[3] = centroids[T] - midpoints[T][3]
    return nothing
end

function edge_lengths!(lengths, T, edges)
    ℓ₁ = norm(edges[1])
    ℓ₂ = norm(edges[2])
    ℓ₃ = norm(edges[3])
    lengths[T] = (ℓ₁, ℓ₂, ℓ₃)
    return nothing
end

function edge_normals!(normals, T, edges, lengths)
    normalised_rotated_e₁_90_cw = [gety(edges[1]), -getx(edges[1])] ./ lengths[T][1]
    normalised_rotated_e₂_90_cw = [gety(edges[2]), -getx(edges[2])] ./ lengths[T][2]
    normalised_rotated_e₃_90_cw = [gety(edges[3]), -getx(edges[3])] ./ lengths[T][3]
    normals[T] = (
        normalised_rotated_e₁_90_cw,
        normalised_rotated_e₂_90_cw,
        normalised_rotated_e₃_90_cw
    )
    return nothing
end

function control_volume_node_centroid!(p, centroids, T, pts)
    p[1] = centroids[T] - pts[1]
    p[2] = centroids[T] - pts[2]
    p[3] = centroids[T] - pts[3]
    return nothing
end

function control_volume_connect_midpoints!(q, midpoints, T)
    q[1] = midpoints[T][1] - midpoints[T][3]
    q[2] = midpoints[T][2] - midpoints[T][1]
    q[3] = midpoints[T][3] - midpoints[T][2]
    return nothing
end

function sub_control_volume_areas!(S, p, q)
    S[1] = 0.5 * abs(p[1][1] * q[1][2] - p[1][2] * q[1][1])
    S[2] = 0.5 * abs(p[2][1] * q[2][2] - p[2][2] * q[2][1])
    S[3] = 0.5 * abs(p[3][1] * q[3][2] - p[3][2] * q[3][1])
    return nothing
end

function shape_function_coefficients!(s, τ, pts)
    p₁, p₂, p₃ = pts
    (x₁, y₁), (x₂, y₂), (x₃, y₃) = p₁, p₂, p₃
    Δ = x₂ * y₃ - y₂ * x₃ - x₁ * y₃ + x₃ * y₁ + x₁ * y₂ - x₂ * y₁
    s[τ] = NTuple{9,number_type(pts)}([
        y₂ - y₃,
        y₃ - y₁,
        y₁ - y₂,
        x₃ - x₂,
        x₁ - x₃,
        x₂ - x₁,
        x₂ * y₃ - x₃ * y₂,
        x₃ * y₁ - x₁ * y₃,
        x₁ * y₂ - x₂ * y₁
    ]) ./ Δ
    return nothing
end

function boundary_information(T::Ts, adj, pts, BNV) where {Ts}
    boundary_edges = boundary_edge_matrix(adj, BNV)
    boundary_normals = outward_normal_boundary(pts, boundary_edges)
    boundary_nodes = [boundary_edges[n][1] for n in eachindex(boundary_edges)]
    V = construct_triangles(Ts)
    Ttype = DelaunayTriangulation.triangle_type(Ts)
    for n in eachindex(boundary_edges)
        i = boundary_edges[n][1]
        j = boundary_edges[n][2]
        k = boundary_edges[n][3]
        τ = construct_triangle(Ttype, i, j, k)
        DelaunayTriangulation.add_triangle!(V, τ)
    end
    boundary_elements = DelaunayTriangulation.remove_duplicate_triangles(V)
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
    return boundary_edges, boundary_normals, boundary_nodes, boundary_elements
end

function boundary_edge_matrix(adj, BNV)
    E = Matrix{Int64}(undef, 4, length(unique(reduce(vcat, BNV)))) #[left;right;adjacent[left,right];type]
    all_boundary_nodes = Vector{Int64}([])
    edge_types = Vector{Int64}([])
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
    E = [[E[1, i], E[2, i], E[3, i], E[4, i]] for i in 1:size(E, 2)]
    return E
end

function outward_normal_boundary(pts, E)
    F = number_type(pts)
    N = Matrix{Union{Vector{F},Int64}}(undef, 2, length(E)) # normal;type 
    for n in eachindex(E)
        v₁ = E[n][1]
        v₂ = E[n][2]
        p = _get_point(pts, v₁)
        q = _get_point(pts, v₂)
        r = q - p
        rx, ry = r
        N[1, n] = [ry, -rx] / norm(r)
        N[2, n] = E[n][4]
    end
    N = [tuple(N[1, n], N[2, n]) for n in eachindex(E)]
    return N
end

function interior_information(boundary_nodes, num_nodes, T::Ts, boundary_elements, DG, pts) where {Ts}
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
    interior_edge_boundary_element_identifier = construct_interior_edge_boundary_element_identifier(boundary_elements, T, DG, pts)
    return interior_nodes, interior_elements, interior_edge_boundary_element_identifier
end

function construct_interior_edge_boundary_element_identifier(boundary_elements, ::Ts, DG, pts) where {Ts}
    V = DelaunayTriangulation.triangle_type(Ts)
    interior_edge_boundary_element_identifier = Dict{V,Vector{NTuple{2,NTuple{2,Int64}}}}()
    sizehint!(interior_edge_boundary_element_identifier, length(boundary_elements))
    idx = convex_hull(DG, pts)
    ch_edges = [(idx[i], idx[i == length(idx) ? 1 : i + 1]) for i in eachindex(idx)]
    for τ in boundary_elements
        v = indices(τ)
        res = Vector{NTuple{2,NTuple{2,Int64}}}()
        sizehint!(res, 2) # # can't have 3 edges in the interior, else it's not a boundary edge
        for (vᵢ, vⱼ) in edges(τ)
            if (vᵢ, vⱼ) ∉ ch_edges # # if this line is an edge of the convex hull, obviously it's not an interior edge
                j₁ = findfirst(vᵢ .== v)
                j₂ = findfirst(vⱼ .== v)
                push!(res, ((vᵢ, j₁), (vⱼ, j₂)))
            end
        end
        interior_edge_boundary_element_identifier[τ] = res
    end
    return interior_edge_boundary_element_identifier
end

##############################################################################
##
## BOUNDARY CONDITIONS 
##
##############################################################################
"""
    BoundaryConditions{BNV,F,Typ,BCP,DN,NN,DuN,INN,BMI,MBI,TM,DuTu,DTu} 

Struct which defines the boundary conditions for a PDE problem.

# Fields 
- `boundary_node_vector::BNV`

Vector of vectors giving the indices for the boundary nodes, separated by boundary type.
- `functions::F`

Collection of functions, giving the boundary conditions for each boundary type. 
The required form for these functions is given below in the description of `type`.
- `type::Typ`

The type of boundary condition for each edge. The following types are permitted:

-- `type[i] = :N`: Means a Neumann condition should be used, taking the form `∇q(x, y, t)⋅n̂(x, y) = functions[i](x, y, p)`. [NOTE: Currently we only allow for this condition to be homogeneous, so the function values will be ignored but do need to be provided.]

-- `type[i] = :D`: Means a Dirichlet condition should be used, taking the form `u(x, y, t) = functions[i](x, y, t, p)`.

-- `type[i] = :dudt`: Means the following time-dependent Dirichlet condition should be used, taking the form `du(x, y, t)/dt = functions[i](x, y, t, u, p)`.

See also [`symbolise_types`](@ref).
- `params::BCP`

Parameters for the `functions`. `params[i]` gives known parameter values for `functions[i]`, 
giving the argument `p` in the above `functions`. If there are no required parameter vectors, 
this can simply be `nothing`.
- `dirichlet_nodes::DN`

This is a vector of indices corresponding to the nodes in the mesh that are of Dirichlet type (`:D`). See also 
[`classify_edges`](@ref).
- `neumann_nodes::NN`

This is a vector of indices corresponding to the nodes in the mesh that are of Neumann type (`:N`). See also 
[`classify_edges`](@ref).
- `dudt_nodes::DuN`

This is a vector of indices corresponding to the nodes in the mesh that are of dudt type (`:dudt`). See also 
[`classify_edges`](@ref).
- `interior_or_neumann_nodes::INN`

This is a vector of indices corresponding to the nodes in the mesh that are either 
interior nodes or Neumann boundary nodes.
- `boundary_to_mesh_idx::BMI`

This gives `Dict(1:num_boundary_nodes .=> boundary_nodes)`, i.e. the mapping between the index of 
nodes around to the boundary to the index of nodes for the whole mesh. See also [`boundary_idx_maps`](@ref).
- `mesh_to_boundary_idx::MBI`

This gives `Dict(boundary_nodes .=> 1:num_boundary_nodes)`, i.e. the mapping between the index of 
nodes in the whole mesh to the index of nodes for the boundary. See also [`boundary_idx_maps`](@ref).
- `type_map::TM`

This is a dictionary such that `type_map[node_idx]` returns the edge type (or `0` for an interior node) 
to use for evaluating the boundary function at node `node_idx`. See also [`construct_type_map`](@ref).
- `dudt_tuples::DuTu`

A tuple of values `(j, type, x, y, p)` for all the `dudt` nodes, defined for the purpose of avoiding type instabilities. 
See [`node_loop_vals`](@ref) for the construction and [`update_dudt_nodes!`](@ref) for its use.
- `dirichlet_tuples::DTu`

A tuple of values `(j, type, x, y, p)` for all the `Dirichlet` nodes, defined for the purpose of avoiding type instabilities. 
See [`node_loop_vals`](@ref) for the construction and [`update_dirichlet_nodes!`](@ref) for its use.

# Constructors 
We provide the following constructor:

    BoundaryConditions(mesh::FVMGeometry, functions, type;
        params=Tuple(Float64[] for _ in 1:length(functions)),
        dirichlet_nodes=nothing, neumann_nodes=nothing, dudt_nodes=nothing, interior_or_neumann_nodes=nothing,
        ∂=nothing, ∂⁻¹=nothing, type_map=nothing)
    
These terms are all as above, with the exception of `mesh` which is the [`FVMGeometry`](@ref).
"""
struct BoundaryConditions{BNV,F,Typ,BCP,DN,NN,DuN,INN,BMI,MBI,TM,DuTu,DTu}
    boundary_node_vector::BNV
    functions::F
    type::Typ
    params::BCP
    dirichlet_nodes::DN
    neumann_nodes::NN
    dudt_nodes::DuN
    interior_or_neumann_nodes::INN
    boundary_to_mesh_idx::BMI
    mesh_to_boundary_idx::MBI
    type_map::TM
    dudt_tuples::DuTu
    dirichlet_tuples::DTu
end
function BoundaryConditions(mesh::FVMGeometry, functions, type;
    params=Tuple(Float64[] for _ in 1:length(functions)),
    dirichlet_nodes=nothing, neumann_nodes=nothing, dudt_nodes=nothing, interior_or_neumann_nodes=nothing,
    ∂=nothing, ∂⁻¹=nothing, type_map=nothing)
    if functions isa Function
        functions = Vector{Function}([functions])
    end
    functions = FunctionWrangler(functions)
    params = Tuple(params)
    type = symbolise_types(type)
    dirichlet_nodes, neumann_nodes, dudt_nodes, interior_or_neumann_nodes = classify_edges(mesh.boundary_edges, type, dirichlet_nodes, neumann_nodes, dudt_nodes, mesh.interior_points, interior_or_neumann_nodes)
    ∂, ∂⁻¹ = boundary_idx_maps(mesh.boundary_edges, ∂, ∂⁻¹)
    type_map = construct_type_map(type_map, dirichlet_nodes, neumann_nodes, dudt_nodes, DIRICHLET_ARGS, NEUMANN_ARGS, DUDT_ARGS, mesh, ∂⁻¹, functions)
    dudt_tuples = node_loop_vals(type_map, params, mesh.points, dudt_nodes)
    dirichlet_tuples = node_loop_vals(type_map, params, mesh.points, dirichlet_nodes)
    return BoundaryConditions(mesh.boundary_node_vector, functions, type, params, dirichlet_nodes,
        neumann_nodes, dudt_nodes, interior_or_neumann_nodes, ∂, ∂⁻¹, type_map,
        dudt_tuples, dirichlet_tuples)
end

"""
    classify_edges(edge_matrix, edge_types)
    classify_edges(edge_matrix, edge_types, dirichlet_nodes, neumann_nodes, dudt_nodes, interior_nodes, interior_or_neumann_nodes)

Classifies the edges of a boundary into Dirichlet, Neumann, or dudt. The second method signature is for 
providing existing values.

# Arguments 
- `edge_matrix`: The edge matrix for the boundary. See also [`construct_boundary_edge_matrix`](@ref).
- `edge_types`: Vector giving the types of each contiguous boundary segment. See also [`symbolise_types`](@ref).

# Outputs
- `dirichlet_nodes`: This is a vector of indices corresponding to the nodes in the mesh that are of Dirichlet type (`:D`).
- `neumann_nodes`: This is a vector of indices corresponding to the nodes in the mesh that are of Neumann type (`:N`).
- `dudt_nodes`: This is a vector of indices corresponding to the nodes in the mesh that are of dudt type (`:dudt`).

# Extended help 
The types are decided by iterating over the boundary edges, looking at the current edge and the edge one before. 
If we let `tᵢ` be the type of the `i`th edge and `tᵢ₋₁` the type of the edge before, then if any of the types 
are Dirichlet we let the `i`th node be a Dirichlet node, then if any of the types are dudt the node is a dudt node, 
and finally if both of the types are Neumann we let the node be a Neumann node. 
"""
function classify_edges end
function classify_edges(edge_matrix, edge_types)
    num_edges = length(edge_matrix)
    dirichlet_nodes = Vector{Int64}([])
    neumann_nodes = Vector{Int64}([])
    dudt_nodes = Vector{Int64}([])
    sizehint!(dirichlet_nodes, num_edges)
    sizehint!(neumann_nodes, num_edges)
    sizehint!(dudt_nodes, num_edges)
    for i in 1:num_edges
        v = edge_matrix[i][1]
        typeᵢ = edge_types[edge_matrix[i][4]]
        typeᵢ₋₁ = edge_types[edge_matrix[i - 1 == 0 ? num_edges : i - 1][4]]
        if typeᵢ == :D || typeᵢ₋₁ == :D# the ith boundary node is associated with the left-most point of the ith edge
            push!(dirichlet_nodes, v)
        elseif typeᵢ == :dudt || typeᵢ₋₁ == :dudt
            push!(dudt_nodes, v)
        elseif typeᵢ == :N && typeᵢ₋₁ == :N
            push!(neumann_nodes, v)
        else
            throw("Incorrect type specification.")
        end
    end
    return dirichlet_nodes, neumann_nodes, dudt_nodes
end
@inline function classify_edges(edge_matrix, edge_types, dirichlet_nodes, neumann_nodes, dudt_nodes, interior_nodes, interior_or_neumann_nodes)
    if isnothing(dirichlet_nodes) || isnothing(neumann_nodes) || isnothing(dudt_nodes)
        dirichlet_nodes, neumann_nodes, dudt_nodes = classify_edges(edge_matrix, edge_types)
    end
    if isnothing(interior_or_neumann_nodes)
        interior_or_neumann_nodes = union(interior_nodes, neumann_nodes)
    end
    return dirichlet_nodes, neumann_nodes, dudt_nodes, interior_or_neumann_nodes
end

"""
    boundary_idx_maps(edge_matrix)
    boundary_idx_maps(edge_matrix, ∂, ∂⁻¹)

Given a edge matrix for the boundary (see also [`construct_boundary_edge_matrix`](@ref)) returns two `Dict`s:

- `∂`: This gives `Dict(1:num_boundary_nodes .=> boundary_nodes)`, i.e. the mapping between the index of nodes around to the boundary to the index of nodes for the whole mesh.
- `∂⁻¹`: This gives `Dict(boundary_nodes .=> 1:num_boundary_nodes)`, i.e. the mapping between the index of nodes in the whole mesh to the index of nodes for the boundary.

If these `Dict`s are provided and non-`nothing`, they are simply returned.
"""
function boundary_idx_maps end
@inline function boundary_idx_maps(edge_matrix)
    num_boundary_nodes = length(edge_matrix)
    boundary_nodes = [edge_matrix[n][1] for n in 1:num_boundary_nodes]
    ∂ = Dict(1:num_boundary_nodes .=> boundary_nodes)
    ∂⁻¹ = Dict(boundary_nodes .=> 1:num_boundary_nodes)
    return ∂, ∂⁻¹
end
@inline function boundary_idx_maps(edge_matrix, ∂, ∂⁻¹)
    if isnothing(∂) || isnothing(∂⁻¹)
        ∂, ∂⁻¹ = boundary_idx_maps(edge_matrix)
    end
    return ∂, ∂⁻¹
end

"""
    check_nargs(F::FunctionWrangler, idx, num_args)

For a given set of functions in `F`, tests if the function in position 
`idx` has a number of arguments `num_args`.
"""
@inline Base.@propagate_inbounds function check_nargs(F::FunctionWrangler, idx, num_args)
    try
        sindex(F, idx, zeros(num_args)...)
    catch
        return false
    end
    return true
end

"""
    construct_type_map!(type_map, nodes, ∂⁻¹, mesh::FVMGeometry, num_args, functions)

Updates the type map dictionary in `type_map` based on the given boundary nodes. This `type_map` 
is updated in-place and no values are returned.

# Arguments 
- `type_map`: The `Dict` to update.
- `nodes`: The boundary nodes to consider.
- `∂⁻¹`: The dictionary `Dict(boundary_nodes .=> 1:num_boundary_nodes)`, i.e. the mapping between the index of nodes in the whole mesh to the index of nodes for the boundary. See also [`boundary_idx_maps`](@ref).
- `mesh::FVMGeometry`: Geometry setup for the FVM. See [`FVMGeometry`](@ref).
- `num_args`: The number of arguments in the functions used for the edges corresponding to `nodes`.
- `functions`: Collection of functions, giving the boundary conditions for each boundary type. Will be converted to `FunctionWrangler` if not already. 
"""
function construct_type_map! end
function construct_type_map!(type_map, nodes, ∂⁻¹, mesh::FVMGeometry, num_args, functions::FunctionWrangler)
    for j ∈ nodes
        ∂j = ∂⁻¹[j]
        type = mesh.boundary_edges[∂j][4]
        if !check_nargs(functions, type, num_args) # If this happens, the previous edge is the cause 
            type = type == 1 ? length(mesh.boundary_node_vector) : type - 1
        end
        type_map[j] = type
    end
    return nothing
end
@inline function construct_type_map!(type_map, nodes, ∂⁻¹, mesh::FVMGeometry, num_args, functions)
    construct_type_map!(type_map, nodes, ∂⁻¹, mesh::FVMGeometry, num_args, FunctionWrangler(functions))
    return nothing
end
"""
    construct_type_map(type_map, nodes₁, nodes₂, nodes₃, args₁, args₂, args₃, mesh::FVMGeometry, ∂⁻¹, functions)

Constructs the type map (see also [`construct_type_map!`](@ref)) for the pairs `(nodesᵢ, argsᵢ)`, `i = 1, 2, 3`. If 
`type_map` is provided and is non-`nothing`, simply returns the `type_map` instead.
"""
@inline function construct_type_map(type_map, nodes₁, nodes₂, nodes₃, args₁, args₂, args₃, mesh::FVMGeometry, ∂⁻¹, functions)
    if isnothing(type_map)
        type_map = Dict{Int64,Int64}([])
        sizehint!(type_map, length(mesh.boundary_edges))
        construct_type_map!(type_map, nodes₁, ∂⁻¹, mesh, args₁, functions)
        construct_type_map!(type_map, nodes₂, ∂⁻¹, mesh, args₂, functions)
        construct_type_map!(type_map, nodes₃, ∂⁻¹, mesh, args₃, functions)
    end
    return type_map
end

"""
    FunctionWrangler(F::FunctionWrangler)

A constructor for `FunctionWrangler` when `F` is already a `FunctionWrangler`. Needed so that we 
can call `FunctionWrangler(F)` in general, whether `F` is a vector of functions or already converted
into a `FunctionWrangler`.
"""
@inline function FunctionWranglers.FunctionWrangler(F::FunctionWrangler)
    return F
end

"""
    symbolise_types(type)

Given a set of boundary condition types in `type`, converts them to the appropriate 
symbolic representations (Neumann is `:N`, Dirichlet is `:D`, and dudt is `:dudt`).
The returned vector is an `SVector`.
"""
@inline Base.@propagate_inbounds function symbolise_types(type)
    if type isa AbstractVector{Symbol}
        return Tuple(type)
    elseif type isa Union{Symbol,String}#not a vector, but length 1 
        return symbolise_types([type])
    elseif type isa NTuple{length(type),Symbol}
        return type
    elseif type isa AbstractVector{String}
        new_type = Vector{Symbol}([])
        for i in eachindex(type)
            if type[i] == "Neumann" || type[i] == "N"
                push!(new_type, :N)
            elseif type[i] == "Dirichlet" || type[i] == "D"
                push!(new_type, :D)
            elseif type[i] == "dudt" || type[i] == "du/dt"
                push!(new_type, :dudt)
            else
                throw("Invalid type.")
            end
        end
        type = NTuple{length(new_type),Symbol}(new_type)
        return type
    else
        throw("Invalid type.")
    end
end

"""
    node_loop_vals(type_map, params, points, nodes)

Constructs a tuple of length `length(nodes)` to help unroll a loop over these nodes.

# Arguments 
- `type_map`: A dictionary from [`construct_type_map`](@ref) that maps node indices to edge numbers.
- `params`: The parameters for the boundary functions. See [`BoundaryConditions`](@ref).
- `points`: Points of the mesh. See [`FVMGeometry`](@ref).
- `nodes`: The boundary nodes that will be looped over. See [`classify_edges`](@ref).

# Outputs 
- `vals`: This is a tuple of length `length(nodes)`, with the `i`th element of the tuple itself a tuple
of the form `(j, type, x, y, p)`, where `j = nodes[i]`, `type = type_map[j]`, `x, y = points[j]`, and 
`p = params[type]`.
"""
@inline Base.@propagate_inbounds function node_loop_vals(type_map, params, points, nodes)
    vals = ntuple(i -> begin
            j = nodes[i]
            type = type_map[j]
            x = getx(DelaunayTriangulation.get_point(points, j))
            y = gety(DelaunayTriangulation.get_point(points, j))
            p = params[type]
            return (j, type, x, y, p)
        end, length(nodes))
    return vals
end

##############################################################################
##
## PROBLEM DEFINITION
##
##############################################################################
@doc raw"""
    FVMProblem{P,T,N,VNV,A,EM,EMa,BNV,V,No,S,Ar,L,M,IN,IE,BEL,BED,BN,BNI,IEBEI,
        F,Typ,BCP,DN,NN,DuN,INN,BMI,MBI,TM,DuTu,DTu,
        Flx,Rct,FlxP,RctP,IC,FT,Slvr,Stdy}

Struct which defines a PDE problem to be solving using the finite volume method, defining a PDE of the form 

```math 
\dfrac{\partial u(x, y, t)}{\partial t} + \boldsymbol\nabla\boldsymbol \cdot \boldsymbol q(x, y, t, u) = R(x, y, t, u),\quad (x, y) \in \Omega,\quad t > 0,\\
u(x, y, 0) = f(x, y), \quad (x, y) \in \Omega, 
```
with appropriate boundary conditions defined according to [`BoundaryConditions`](@ref). We describe this mathematical
specification some more later under the constructor heading.

# Fields 
- `mesh::FVMGeometry{P,T,N,VNV,A,EM,EMa,BNV,V,No,S,Ar,L,M,IN,IE,BEL,BED,BN,BNI,BNE}`

Geometry setup for the FVM. See [`FVMGeometry`](@ref).
- `boundary_conditions::BoundaryConditions{BNV,F,Typ,BCP,DN,NN,DuN,INN,BMI,MBI,TM,DuTu,DTu}`

Boundary conditions for the FVM. See [`BoundaryConditions`](@ref).
- `flux!::Flx`

Flux function, taking the form `flux!(q, x, y, t, u, p)` for some known parameters `p = flux_parameters`. The function should compute the flux `q` in-place, updated into `q`.
- `reaction::Rct`

Reaction function, taking the form `R(x, y, t, u, p)` for some known parameters `p = reaction_parameters`.
- `flux_parameters::FlxP`

Known parameters for the flux function, given as the last argument `p` in `flux!`. If there are no such parameters, this field can simply be `nothing`.
- `reaction_parameters::RctP`

Known parameters for the reaction function, given as the last argument `p` in `reaction`. If there are no such parameters, this field can simply be `nothing`.
- `initial_condition::IC`

Initial condition to be used, where `initial_condition[i]` is the value of `u` at `t = 0` and at `(x, y) = mesh.DT.points[i]`.
- `final_time::FT`

Time to solve the problem up to, giving a timespan of `(0, final_time)`.
- `solver::Slvr`

Solver to use for solving the resulting system of ODEs with `DifferentialEquations.jl`.
- `steady::Stdy`

Whether the problem is a steady problem (`steady = true`) or not (`steady = false`).

# Constructors and Mathematical Specification

We provide the following constructor:

    FVMProblem(; 
        mesh, boundary_conditions,
        delay=nothing, delay_parameters=nothing, diffusion=nothing, diffusion_parameters=nothing,
        reaction=nothing, reaction_parameters=nothing, flux! = nothing, flux_parameters=nothing,
        initial_condition, final_time, solver=nothing, steady=false) 

These terms are as above, except we also provide the following arguments:

- `delay`.
- `delay_parameters`.
- `diffusion`.
- `diffusion_parameters`.

These arguments can be used in place of a `flux!` function (along with a reaction function), for defining 
the delay-reaction-diffusion problem 
```math
\dfrac{\partial u(x, y, t)}{\partial t} = T(t; \boldsymbol \alpha) \left[\boldsymbol \nabla \boldsymbol \cdot \left(D(x, y, t, u; \boldsymbol \beta) \boldsymbol \nabla u(x, y, t)\right) + R(x, y, t, u; \boldsymbol \gamma)\right],
```
where ``T(t; \boldsymbol \alpha)`` is the `delay` function with ``\alpha`` given by `delay_parameters`, and 
similarly for the `diffusion` function ``D(\boldsymbol x, t, u; \boldsymbol \beta)``; the `reaction_parameters` above 
are ``\boldsymbol \gamma``. When these functions are provided (and no `flux!` provided), then we construct 
the corresponding `flux!` and `reaction` functions, noting that in the above formulation 

```math
\boldsymbol q(x, y, t, u) = -T(t; \boldsymbol \alpha)D(x, y, t, u; \boldsymbol \beta)\nabla u(\boldsymbol x, t)
```
and  
```math 
\tilde R(x, y, t, u) = T(t; \boldsymbol \alpha)R(x, y, t, u; \boldsymbol \gamma).
```
See also [`return_flux_fnc`](@ref) and [`return_reaction_fnc`](@ref).
"""
struct FVMProblem{iip_flux,FG<:FVMGeometry,BC<:BoundaryConditions,Flx,Rct,FlxP,RctP,IC,FT,Slvr,Stdy}
    mesh::FG
    boundary_conditions::BC
    flux!::Flx
    reaction::Rct
    flux_parameters::FlxP
    reaction_parameters::RctP
    initial_condition::IC
    initial_time::FT
    final_time::FT
    solver::Slvr
    steady::Stdy
end
function FVMProblem(;
    mesh, boundary_conditions, iip_flux=true,
    delay=nothing, delay_parameters=nothing, diffusion=nothing, diffusion_parameters=nothing,
    reaction=nothing, reaction_parameters=nothing, (flux!)=nothing, flux_parameters=nothing,
    initial_condition, initial_time=0.0, final_time, solver=Tsit5(), steady=false)
    flux_fnc!, flux_params = return_flux_fnc(flux!, flux_parameters, delay, delay_parameters, diffusion, diffusion_parameters; iip_flux)
    reaction_fnc, reac_params = return_reaction_fnc(delay, delay_parameters, reaction_parameters, reaction)
    return FVMProblem{iip_flux,typeof(mesh),typeof(boundary_conditions),typeof(flux_fnc!),
        typeof(reaction_fnc),typeof(flux_params),typeof(reac_params),typeof(initial_condition),
        typeof(final_time),typeof(solver),typeof(steady)}(mesh, boundary_conditions, flux_fnc!,
        reaction_fnc, flux_params, reac_params, initial_condition, initial_time, final_time, solver, steady)
end
SciMLBase.isinplace(::FVMProblem{iip_flux,FG,BC,
    Flx,Rct,FlxP,RctP,IC,FT,Slvr,Stdy}) where {iip_flux,FG,BC,
    Flx,Rct,FlxP,RctP,IC,FT,Slvr,Stdy} = iip_flux

"""
    return_flux_fnc(flux!, flux_parameters, delay, delay_parameters, diffusion, diffusion_parameters)

Computes the object for the flux function in case none is provided. See [`FVMProblem`](@ref).

# Arguments 
- `flux!`: Either `nothing` if no flux function has been provided, or a function if it should be returned with no action taken.
- `flux_parameters`: Parameters for the flux function; if `flux! = nothing`, this argument is not used.
- `delay`: The delay function.
- `delay_parameters`: Parameters for the delay function.
- `diffusion`: The diffusion function. 
- `diffusion_parameters`: Parameters for the diffusion function. 

# Outputs 
- `flux_fnc!`: The flux function. 
- `flux_params`: The parameters for the flux function.
"""
function return_flux_fnc(flux!, flux_parameters, delay, delay_parameters, diffusion, diffusion_parameters; iip_flux=true)
    if isnothing(flux!)
        if !isnothing(delay)
            if iip_flux
                flux_fnc! = @inline (q, x, y, t, α, β, γ, p) -> begin
                    #local delay_parameters, diffusion_parameters, delay, diffusion # https://discourse.julialang.org/t/extra-allocations-time-when-a-function-is-returned-from-another-function-than-defined-directly/81933/5
                    #delay_parameters, diffusion_parameters, delay, diffusion = p
                    local delay_parameters, diffusion_parameters
                    delay_parameters, diffusion_parameters = p
                    q[1] = -delay(t, delay_parameters) * diffusion(x, y, t, α * x + β * y + γ, diffusion_parameters) * α
                    q[2] = -delay(t, delay_parameters) * diffusion(x, y, t, α * x + β * y + γ, diffusion_parameters) * β
                    return nothing
                end
            else
                flux_fnc! = @inline ((x, y, t, α::T, β::T, γ::T, p) where {T}) -> begin
                    local delay_parameters, diffusion_parameters
                    delay_parameters, diffusion_parameters = p
                    q = [-delay(t, delay_parameters) * diffusion(x, y, t, α * x + β * y + γ, diffusion_parameters) * α,
                        -delay(t, delay_parameters) * diffusion(x, y, t, α * x + β * y + γ, diffusion_parameters) * β]
                    return q
                end
            end
            flux_params = [delay_parameters, diffusion_parameters]
            return flux_fnc!, flux_params
        else
            if iip_flux
                flux_fnc! = @inline (q, x, y, t, α, β, γ, p) -> begin
                    local diffusion_parameters
                    diffusion_parameters = p
                    q[1] = -diffusion(x, y, t, α * x + β * y + γ, diffusion_parameters) * α
                    q[2] = -diffusion(x, y, t, α * x + β * y + γ, diffusion_parameters) * β
                    return nothing
                end
            else
                flux_fnc! = @inline ((x, y, t, α::T, β::T, γ::T, p) where {T}) -> begin
                    local diffusion_parameters
                    diffusion_parameters = p
                    q = [-diffusion(x, y, t, α * x + β * y + γ, diffusion_parameters) * α,
                        -diffusion(x, y, t, α * x + β * y + γ, diffusion_parameters) * β]
                    return q
                end
            end
            flux_params = deepcopy(diffusion_parameters)
            return flux_fnc!, flux_params
        end
    else
        return flux!, flux_parameters
    end
end

"""
    return_reaction_fnc(delay, delay_parameters, reaction_parameters, reaction)

Computes the object for the reaction function in case a delay function is provided. See [`FVMProblem`](@ref).

# Arguments 
- `delay`: The delay function.
- `delay_parameters`: Parameters for the delay function.
- `reaction_parameters`: Parameters for the reaction function. 
- `reaction`: The reaction function. 

# Outputs 
- `reac_fnc`: The updated reaction function. 
- `reac_params`: The parameters for the updated reaction function.
"""
function return_reaction_fnc(delay, delay_parameters, reaction_parameters, reaction)
    if !isnothing(delay)
        reaction_fnc = (x, y, t, u, p) -> begin
            #local delay_parameters, reaction_parameters, delay, reaction # https://discourse.julialang.org/t/extra-allocations-time-when-a-function-is-returned-from-another-function-than-defined-directly/81933/5
            #delay_parameters, reaction_parameters, delay, reaction = p
            local delay_parameters, reaction_parameters = p
            return delay(t, delay_parameters) * reaction(x, y, t, u, reaction_parameters)
        end
        # reac_params = (delay_parameters, reaction_parameters, delay, reaction)
        reac_params = [delay_parameters, reaction_parameters]
        return reaction_fnc, reac_params
    else
        return reaction, reaction_parameters
    end
end

# some methods for extracting components
@inline Base.@propagate_inbounds gets(prob::FVMProblem, k, i) = prob.mesh.shape_function_coeffs[k][i]
@inline Base.@propagate_inbounds getmidpoint(prob::FVMProblem, k, i) = prob.mesh.midpoints[k][i]
@inline Base.@propagate_inbounds getnormal(prob::FVMProblem, k, i) = prob.mesh.normals[k][i]
@inline Base.@propagate_inbounds getlength(prob::FVMProblem, k, i) = prob.mesh.lengths[k][i]
@inline Base.@propagate_inbounds getxy(prob::FVMProblem, i) = _get_point(prob.mesh.points, i)
@inline Base.@propagate_inbounds getvolume(prob::FVMProblem, i) = prob.mesh.volumes[i]
@inline Base.@propagate_inbounds identify_interior_edge_boundary_element(prob::FVMProblem, v) = interior_edge_boundary_element_identifier(prob.mesh, v)
@inline getreaction(prob::FVMProblem, x, y, t, u, reaction_parameters) = prob.reaction(x, y, t, u, reaction_parameters)
@inline function getflux!(prob::FVMProblem{iip_flux,FG,BC,
        Flx,Rct,FlxP,RctP,IC,FT,Slvr,Stdy}, flux_vec, x, y, t, α, β, γ, flux_params) where {iip_flux,FG,BC,
    Flx,Rct,FlxP,RctP,IC,FT,Slvr,Stdy}
    if iip_flux
        prob.flux!(flux_vec, x, y, t, α, β, γ, flux_params)
        return nothing
    else
        q = prob.flux!(x, y, t, α, β, γ, flux_params)
        return q
    end
end

##############################################################################
##
## PARAMETERS
##
##############################################################################
"""
    mutable struct FVMParameters{MPT,NRM,LEN,FLXV,SHC,FLXP,S,INN,IEBEI,RCT,RCTP,PTS,VOLS,DUDTT,BCF,TRIV,INTELE,BNDELE}

Struct for the FVM parameters for an [`FVMProblem`].

# Fields 
- `const midpoints::MPT` 

Midpoints for each control volume edge. See also [`midpoint!`](@ref).
- `const normals::NRM` 

Normals of the edges in the individual control volumes for the mesh. See also [`edge_normals!`](@ref).
- `const lengths::LEN` 

Lengths for each control volume edge. See also [`edge_lengths!`](@ref).
- `const flux!::FLX`

Flux function, taking the form `flux!(q, x, y, t, u, p)` for some known parameters `p = flux_params`. The function should compute the flux `q` in-place, updated into `q`.
- `flux_vec::FLXV`

A cache vector `zeros(2)` that gets updated in-place with the flux vector.
- `shape_coeffs::SHC` 

Cache array for the coefficients for the linear shape function over the `k`th element.
- `flux_params::FLXP` 

Known parameters for the flux function, given as the last argument `p` in `flux!`. If there are no such parameters, this field can simply be `nothing`.
- `const s::S `

Shape function coefficients for the linear shape functions in the mesh. See also [`shape_function_coefficients!`](ref).
- `const interior_or_neumann_nodes::INN` 

This is a vector of indices corresponding to the nodes in the mesh that are either interior nodes or Neumann boundary nodes.
- `const int_edge_bnd_el_id::IEBEI` 

Given a set of vertices for a boundary element, identifies the edges corresponding to interior edges. See also [`construct_interior_edge_boundary_element_identifier`](@ref).
- `const react::RCT`

Reaction function, taking the form `react(x, y, t, u, p)` for some known parameters `p = react_parameters`. Known parameters for the reaction function, given as the last argument `p` in `react`. If there are no such parameters, this field can simply be `nothing`.
- `react_params::RCTP` 

Known parameters for the reaction function, given as the last argument `p` in `react`. If there are no such parameters, this field can simply be `nothing`.
- `const points::PTS` 

Points of the mesh.
- `const volumes::VOLS` 

Volumes of the individual control volumes for the mesh.
- `const dudt_tuples::DUDTT` 

A tuple of values `(j, type, x, y, p)` for all the `dudt` nodes, defined for the purpose of avoiding type instabilities. 
- `const bc_functions::BCF` 

Collection of functions, giving the boundary conditions for each boundary type. 
- `const tri_vertices::TRIV` 

A dictionary mapping triangular elements to their vertices, with triangles indexed by the number in the mesh.
- `const interior_elements::INTELE` 

Indices for the triangles of the mesh that do not share a mesh with the boundary.
- `const boundary_elements::BNDELE` 

Indices for the triangles of the mesh that share an edge with the boundary. See also [`boundary_information`](@ref).

# Constructors 
We provide the construct 

    FVMParameters(prob::FVMProblem; cache_eltype = eltype(prob.initial_condition)).

You can change the eltype of the cache arrays `flux_vec` and `shape_coeffs` (see below) using the keyword argument 
`cache_eltype`.
"""
mutable struct FVMParameters{IIP,MPT,NRM,LEN,FLX,FLXV,SHC,FLXP,S,INN,IEBEI,RCT,RCTP,PTS,VOLS,DUDTT,BCF,INTELE,BNDELE}
    const midpoints::MPT
    const normals::NRM
    const lengths::LEN
    const flux!::FLX
    const flux_vec::FLXV
    const shape_coeffs::SHC
    flux_params::FLXP
    const s::S
    const interior_or_neumann_nodes::INN
    const int_edge_bnd_el_id::IEBEI
    const react::RCT
    react_params::RCTP
    const points::PTS
    const volumes::VOLS
    const dudt_tuples::DUDTT
    const bc_functions::BCF
    const interior_elements::INTELE
    const boundary_elements::BNDELE
    const iip_flux::IIP
end
function FVMParameters(prob::FVMProblem{iip_flux,FG,BC,
        Flx,Rct,FlxP,RctP,IC,FT,Slvr,Stdy}; cache_eltype=eltype(prob.initial_condition), parallel=false) where {iip_flux,FG,BC,
    Flx,Rct,FlxP,RctP,IC,FT,Slvr,Stdy}
    p = (midpoints=prob.mesh.midpoints,
        normals=prob.mesh.normals,
        lengths=prob.mesh.lengths,
        (flux!)=prob.flux!,
        flux_vec=!parallel ? dualcache(zeros(cache_eltype, 2), 12) : dualcache(zeros(cache_eltype, 2, num_triangles(prob.mesh), 3), 12),
        shape_coeffs=!parallel ? dualcache(zeros(cache_eltype, 3), 12) : dualcache(zeros(cache_eltype, 3, num_triangles(prob.mesh)), 12),
        flux_params=prob.flux_parameters,
        s=prob.mesh.shape_function_coeffs,
        interior_or_neumann_nodes=prob.boundary_conditions.interior_or_neumann_nodes,
        int_edge_bnd_el_id=prob.mesh.interior_edge_boundary_element_identifier,
        react=prob.reaction,
        react_params=prob.reaction_parameters,
        points=prob.mesh.points,
        volumes=prob.mesh.volumes,
        dudt_tuples=prob.boundary_conditions.dudt_tuples,
        bc_functions=prob.boundary_conditions.functions,
        interior_elements=prob.mesh.interior_elements,
        boundary_elements=prob.mesh.boundary_elements,
        iip_flux=iip_flux)
    return FVMParameters(p...)
end

##############################################################################
##
## FVM EQUATIONS 
##
##############################################################################
"""
    linear_shape_function_coefficients!(shape_coeffs, u, v, s, k)

Computes the linear shape function coefficients `(αₖ, βₖ, γₖ)` for the solution values `u` 
and nodal indices `v` over the `k`th element, with `s` coefficients stored in `s`. These coefficients 
are written into `shape_coeffs`.
"""
@inline Base.@propagate_inbounds function linear_shape_function_coefficients!(shape_coeffs, u, s, T)
    i, j, k = indices(T)
    shape_coeffs[1] = s[T][1] * u[i] + s[T][2] * u[j] + s[T][3] * u[k]
    shape_coeffs[2] = s[T][4] * u[i] + s[T][5] * u[j] + s[T][6] * u[k]
    shape_coeffs[3] = s[T][7] * u[i] + s[T][8] * u[j] + s[T][9] * u[k]
    return nothing
end
@inline Base.@propagate_inbounds function linear_shape_function_coefficients(u::AbstractVector, s, T)
    i, j, k = indices(T)
    shape_coeffs = (s[T][1] * u[i] + s[T][2] * u[j] + s[T][3] * u[k],
        s[T][4] * u[i] + s[T][5] * u[j] + s[T][6] * u[k],
        s[T][7] * u[i] + s[T][8] * u[j] + s[T][9] * u[k])
    return shape_coeffs
end

"""
    fvm_eqs_edge!(du, t, midpoints, normals, lengths, flux!, flux_vec, v, k[, j, jnb], αₖ, βₖ, γₖ, flux_params, interior_or_neumann_nodes)

Computes the summands for the FVM equation over all edges of the `k`th element, adding them to `du`. If 
`(j, jnb)` are provided for the `j`th edge (with `jnb = (j % 3) + 1` the shared vertex), computes the 
summand for that edge.

# Arguments 
- `du`: The current values of `du/dt`. This will be updated in-place.
- `t`: The current time.
- `midpoints`: Midpoints for each control volume edge. See also [`midpoint!`](@ref).
- `normals`: Normals of the edges in the individual control volumes for the mesh. See also [`edge_normals!`](@ref).
- `lengths`: Lengths for each control volume edge. See also [`edge_lengths!`](@ref).
- `flux!`: Flux function, taking the form `flux!(q, x, y, t, u, p)` for some known parameters `p = flux_params`. The function should compute the flux `q` in-place, updated into `q`.
- `flux_vec`: A cache vector `zeros(2)` that gets updated in-place with the flux vector.
- `v`: The nodal indices for the `k`th element.
- `k`: The index of the element.
- `j`: The edge to compute over.
- `jnb`: The other edge shared by `v[j]`.
- `αₖ, βₖ, γₖ`: Coefficients for the linear shape function. See [`linear_shape_function_coefficients!`](@ref).
- `flux_params`: Parameters for the `flux` function.
- `interior_or_neumann_nodes`: This is a vector of indices corresponding to the nodes in the mesh that are either interior nodes or Neumann boundary nodes.

# Outputs 
There are no outputs; `du` is updated in-place with the incremented values.
"""
function fvm_eqs_edge! end
@inline Base.@propagate_inbounds function fvm_eqs_edge!(du, t, midpoints, normals, lengths, flux!,
    flux_vec, T, (vj, j), (vjnb, jnb), αₖ, βₖ, γₖ, flux_params, interior_or_neumann_nodes, iip_flux)
    x, y = midpoints[T][j]
    xn, yn = normals[T][j]
    ℓ = lengths[T][j]
    if iip_flux
        flux!(flux_vec, x, y, t, αₖ, βₖ, γₖ, flux_params)
        summand = -(flux_vec[1] * xn + flux_vec[2] * yn) * ℓ
    else
        q = flux!(x, y, t, αₖ, βₖ, γₖ, flux_params)
        summand = -(q[1] * xn + q[2] * yn) * ℓ
    end
    du[vj] = du[vj] + summand * (vj ∈ interior_or_neumann_nodes)
    du[vjnb] = du[vjnb] - summand * (vjnb ∈ interior_or_neumann_nodes)
    return nothing
end
@inline Base.@propagate_inbounds function fvm_eqs_edge!(du, t, midpoints, normals, lengths, flux!,
    flux_vec, T, αₖ, βₖ, γₖ, flux_params, interior_or_neumann_nodes, iip_flux)#jnb = (j % 3) + 1 # Control volume edge j (shared by vertex jnb)
    i, j, k = indices(T)
    fvm_eqs_edge!(du, t, midpoints, normals, lengths, flux!, flux_vec, T, (i, 1), (j, 2), αₖ, βₖ, γₖ, flux_params, interior_or_neumann_nodes, iip_flux)#unrolled
    fvm_eqs_edge!(du, t, midpoints, normals, lengths, flux!, flux_vec, T, (j, 2), (k, 3), αₖ, βₖ, γₖ, flux_params, interior_or_neumann_nodes, iip_flux)
    fvm_eqs_edge!(du, t, midpoints, normals, lengths, flux!, flux_vec, T, (k, 3), (i, 1), αₖ, βₖ, γₖ, flux_params, interior_or_neumann_nodes, iip_flux)
    return nothing
end

"""
    fvm_eqs_interior_element!(du, u, t, midpoints, normals, lengths, s, flux!, flux_vec[, k], shape_coeffs, flux_params, interior_or_neumann_nodes, interior_elements, tri_vertices)

Computes the summands for the FVM equation over all elements, adding them to `du`. If 
`k` is provided, only computes the summands over the `k`th element.

# Arguments 
- `du`: The current values of `du/dt`. This will be updated in-place.
- `u`: The current values of solution.
- `t`: The current time.
- `midpoints`: Midpoints for each control volume edge. See also [`midpoint!`](@ref).
- `normals`: Normals of the edges in the individual control volumes for the mesh. See also [`edge_normals!`](@ref).
- `lengths`: Lengths for each control volume edge. See also [`edge_lengths!`](@ref).
- `s`: Shape function coefficients for the linear shape functions in the mesh. See also [`shape_function_coefficients!`](ref).
- `flux!`: Flux function, taking the form `flux!(q, x, y, t, u, p)` for some known parameters `p = flux_params`. The function should compute the flux `q` in-place, updated into `q`.
- `flux_vec`: A cache vector `zeros(2)` that gets updated in-place with the flux vector.
- `k`: The index of the element.
- `shape_coeffs`: Coefficients for the linear shape function. See [`linear_shape_function_coefficients!`](@ref).
- `flux_params`: Parameters for the `flux` function.
- `interior_or_neumann_nodes`: This is a vector of indices corresponding to the nodes in the mesh that are either interior nodes or Neumann boundary nodes.
- `interior_elements`: Indices of the interior nodes of the mesh.
- `tri_vertices`: A `Dict` such that `tri_vertices[k]` gives the vertices of the `k`th triangular element in the mesh.

# Outputs 
There are no outputs; `du` is updated in-place with the incremented values.
"""
function fvm_eqs_interior_element! end
@inline Base.@propagate_inbounds function fvm_eqs_interior_element!(du, u, t, midpoints, normals, lengths, s, flux!,
    flux_vec, T, shape_coeffs, flux_params, interior_or_neumann_nodes, interior_elements, iip_flux)
    linear_shape_function_coefficients!(shape_coeffs, u, s, T)
    fvm_eqs_edge!(du, t, midpoints, normals, lengths, flux!, flux_vec, T,
        shape_coeffs[1], shape_coeffs[2], shape_coeffs[3], flux_params, interior_or_neumann_nodes, iip_flux)
    return nothing
end
@inline Base.@propagate_inbounds function fvm_eqs_interior_element!(du, u, t, midpoints, normals, lengths, s, flux!,
    flux_vec, shape_coeffs, flux_params, interior_or_neumann_nodes, interior_elements, iip_flux)
    for T in interior_elements
        fvm_eqs_interior_element!(du, u, t, midpoints, normals, lengths, s, flux!,
            flux_vec, T, shape_coeffs, flux_params, interior_or_neumann_nodes, interior_elements, iip_flux)
    end
    return nothing
end

"""
    fvm_eqs_boundary_element!(du, u, t, midpoints, normals, lengths, flux!, flux_vec[, k], shape_coeffs, flux_params, interior_or_neumann_nodes, int_edge_bnd_el_id, boundary_elements, tri_vertices, s)

Computes the FVM equations over all element which have an edge on the boundary, or only for the 
`k`th such element if `k` is provided.

# Arguments 
- `du`: The current values of `du/dt`. This will be updated in-place.
- `u`: The current values of solution.
- `t`: The current time.
- `midpoints`: Midpoints for each control volume edge. See also [`midpoint!`](@ref).
- `normals`: Normals of the edges in the individual control volumes for the mesh. See also [`edge_normals!`](@ref).
- `lengths`: Lengths for each control volume edge. See also [`edge_lengths!`](@ref).
- `flux!`: Flux function, taking the form `flux!(q, x, y, t, u, p)` for some known parameters `p = flux_params`. The function should compute the flux `q` in-place, updated into `q`.
- `flux_vec`: A cache vector `zeros(2)` that gets updated in-place with the flux vector.
- `k`: The index of the boundary element.
- `shape_coeffs`: Cache array for the coefficients for the linear shape function over the `k`th element
- `flux_params`: Parameters for the flux fnuction.
- `interior_or_neumann_nodes`: This is a vector of indices corresponding to the nodes in the mesh that are either interior nodes or Neumann boundary nodes.
- `int_edge_bnd_el_id`: Given a set of vertices for a boundary element, identifies the edges corresponding to interior edges. See also [`construct_interior_edge_boundary_element_identifier`](@ref).
- `boundary_elements`: Indices for the triangles of the mesh that share an edge with the boundary. See also [`boundary_information`](@ref).
- `tri_vertices`: A `Dict` such that `tri_vertices[k]` gives the vertices of the `k`th triangular element in the mesh.
- `s`: Shape function coefficients for the linear shape functions in the mesh. See also [`shape_function_coefficients!`](ref).

# Outputs 
There are no outputs; `du` is updated in-place with the incremented values.
"""
function fvm_eqs_boundary_element! end
@inline Base.@propagate_inbounds function fvm_eqs_boundary_element!(du, u, t, midpoints, normals, lengths, flux!,
    flux_vec, T, shape_coeffs, flux_params, interior_or_neumann_nodes, int_edge_bnd_el_id, boundary_elements, s, iip_flux)
    linear_shape_function_coefficients!(shape_coeffs, u, s, T)
    interior_edges = int_edge_bnd_el_id[T]
    for (j, jnb) in interior_edges
        fvm_eqs_edge!(du, t, midpoints, normals, lengths, flux!, flux_vec, T, j, jnb, shape_coeffs[1], shape_coeffs[2], shape_coeffs[3], flux_params, interior_or_neumann_nodes, iip_flux)
    end
    return nothing
end
@inline function fvm_eqs_boundary_element!(du, u, t, midpoints, normals, lengths, flux!,#interestingly, NOT using inline here creates 1122 allocations per run on an example ?????????????????????????????
    flux_vec, shape_coeffs, flux_params, interior_or_neumann_nodes, int_edge_bnd_el_id, boundary_elements, s, iip_flux)
    for T in boundary_elements
        fvm_eqs_boundary_element!(du, u, t, midpoints, normals, lengths, flux!,
            flux_vec, T, shape_coeffs, flux_params, interior_or_neumann_nodes, int_edge_bnd_el_id, boundary_elements, s, iip_flux)
    end
    return nothing
end

"""
    fvm_eqs_source_contribution!(du, u, t[, j], react, react_params, points, volumes, interior_or_neumann_nodes)

Given the computed summads for `du` now updated into `du`, computes the complete equation 
`du = du/V + R` at the `j`th node and at time `t`. If `j` is not provided, updates all nodes.

# Arguments 
- `du`: The current values of `du/dt`. This will be updated in-place.
- `u`: The current values of solution.
- `t`: The current time.
- `j`: The node to update.
- `react`: Reaction function, taking the form `react(x, y, t, u, p)` for some known parameters `p = react_parameters`.
- `react_params`: The parameters for the reaction function.
- `points`: Points of the mesh.
- `volumes`: Volumes of the individual control volumes for the mesh.
- `interior_or_neumann_nodes`: This is a vector of indices corresponding to the nodes in the mesh that are either interior nodes or Neumann boundary nodes.

# Outputs 
There are no outputs, but `du[j]` is updated in-place with `du[j] = du[j]/V + R`.
"""
function fvm_eqs_source_contribution! end
@inline Base.@propagate_inbounds function fvm_eqs_source_contribution!(du, u, t, j, react, react_params, points, volumes, interior_or_neumann_nodes)
    x, y = _get_point(points, j)
    V = volumes[j]
    R = react(x, y, t, u[j], react_params)
    du[j] = 1 / V * du[j] + R
    return nothing
end
@inline Base.@propagate_inbounds function fvm_eqs_source_contribution!(du, u, t, react, react_params, points, volumes, interior_or_neumann_nodes)
    for j ∈ interior_or_neumann_nodes
        fvm_eqs_source_contribution!(du, u, t, j, react, react_params, points, volumes, interior_or_neumann_nodes)
    end
    return nothing
end

"""
    update_dudt_nodes!(du, u, t, vals, F)
    update_dudt_nodes!(du, u, t, vals::Tuple{}, F)

Updates the nodes with a `dudt` boundary condition. The latter method signature is a 
fallback method in the case that there are no such nodes.

# Arguments 
- `du`: The values of `du` that get updated with the boundary condition value.
- `u`: The current solution values.
- `t`: The time.
- `vals`: This is a tuple from [`node_loop_vals`](@ref) for the `dudt` nodes.
- `F`: The boundary condition functions. See [`BoundaryConditions`](@ref).

# Outputs 
There are no outputs; `du` is updated in-place with the new values.
"""
function update_dudt_nodes! end
@unroll function update_dudt_nodes!(du, u, t, vals, F)
    @unroll for jtxyp in vals
        j, type, x, y, p = jtxyp
        du[j] = sindex(F, type, x, y, t, u[j], p)
    end
    return nothing
end
@inline function update_dudt_nodes!(du, u, t, vals::Tuple{}, F)
    return nothing
end

"""
    fvm_eqs!(du, u, p, t)

Computes the value of `du/dt` at the nodes of a mesh according to the problem defined in 
`prob`. See [`construct_fvm_parameters`](@ref) for the definition of `p`. See also 
[`CommonSolve.solve(::FVMProblem)`](@ref).
"""
@inline Base.@propagate_inbounds function fvm_eqs!(du::AbstractVector{T}, u, p, t) where {T}
    # prob, q, αβγₖ = p
    (; midpoints, normals, lengths, flux!, flux_vec, shape_coeffs, flux_params, s,
        interior_or_neumann_nodes, int_edge_bnd_el_id, react, react_params, points,
        volumes, dudt_tuples, bc_functions, interior_elements, boundary_elements, iip_flux) = p
    flux_vec = get_tmp(flux_vec, u)
    shape_coeffs = get_tmp(shape_coeffs, u)
    fill!(du, zero(T))
    fvm_eqs_interior_element!(du, u, t, midpoints, normals, lengths, s, flux!,
        flux_vec, shape_coeffs, flux_params, interior_or_neumann_nodes, interior_elements, iip_flux)
    fvm_eqs_boundary_element!(du, u, t, midpoints, normals, lengths, flux!,
        flux_vec, shape_coeffs, flux_params, interior_or_neumann_nodes, int_edge_bnd_el_id, boundary_elements, s, iip_flux)
    fvm_eqs_source_contribution!(du, u, t, react, react_params, points, volumes, interior_or_neumann_nodes)
    update_dudt_nodes!(du, u, t, dudt_tuples, bc_functions)
    return nothing
end

"""
    update_dirichlet_nodes!(u, t, dirichlet_tuples, bc_functions)
    update_dirichlet_nodes!(integrator, dirichlet_tuples, bc_functions)

This function is used for the callback in [`CommonSolve.solve(::FVMProblem)`](@ref) for updating 
the Dirichlet nodes. In particular, it is the `affect!` function. 

# Arguments 
- `u`: The current solution vals, given from `integrator.u`.
- `t`: The current time, given from `integrator.t`.
- `dirichlet_tuples`: This is a tuple from [`node_loop_vals`](@ref) for the Dirichlet nodes.
- `bc_functions`: Set of boundary functions, given in a `FunctionWrangler`.
- `integrator`: The ODE integrator.

# Outputs 
There are no outputs, but `integrator` gets updated in-place with the updated solution values.
"""
function update_dirichlet_nodes! end
@unroll function update_dirichlet_nodes!(u, t, dirichlet_tuples, bc_functions)
    @unroll for jtxyp in dirichlet_tuples
        j, type, x, y, p = jtxyp
        u[j] = sindex(bc_functions, type, x, y, t, p)
    end
    return nothing
end
@inline Base.@propagate_inbounds function update_dirichlet_nodes!(integrator, dirichlet_tuples, bc_functions)
    update_dirichlet_nodes!(integrator.u, integrator.t, dirichlet_tuples, bc_functions)
    return nothing
end

"""
    dirichlet_callback(prob::FVMProblem; no_saveat=false)
    dirichlet_callback(dirichlet_tuples, bc_functions; no_saveat = false)

Function for building the `DiscreteCallback` for the [`FVMProblem`](@ref) `prob` for updating 
the Dirichlet nodes. This callback is the only returned value.

The keyword argument `no_saveat=false` keyword argument is needed to set the `save_positions` keyword 
argument of the `DiscreteCallback`, since it can bug out otherwise when we want to save the solution 
at specific times. See https://discourse.julialang.org/t/avoiding-plotting-to-dense-plots-saveat-not-working/9540/2.
"""
function dirichlet_callback end
@inline function dirichlet_callback(prob::FVMProblem; no_saveat=false)
    return dirichlet_callback(prob.boundary_conditions.dirichlet_tuples, prob.boundary_conditions.functions; no_saveat)
end
@inline function dirichlet_callback(dirichlet_tuples, bc_functions; no_saveat=false)
    condition = (u, t, integrator) -> true
    affect_function = @closure integrator -> begin
        update_dirichlet_nodes!(integrator, dirichlet_tuples, bc_functions)
        return nothing
    end
    cb = DiffEqBase.DiscreteCallback(condition, affect_function; save_positions=(no_saveat, no_saveat))
    return cb
end

##############################################################################
##
## SOLVER
##
##############################################################################
"""
    Symbolics.jacobian_sparsity(prob::FVMProblem)

Generate the sparsity pattern for a given [`FVMProblem`](@ref) `prob`.
"""
function jacobian_sparsity(prob::FVMProblem)
    DG = deepcopy(prob.mesh.neighbours)
    DelaunayTriangulation.delete_point!(DG, DelaunayTriangulation.BoundaryIndex)
    jac_sparsity = adjacency(DG.graph) + I
    return sparse(jac_sparsity)
end

"""
    CommonSolve.solve(prob::FVMProblem[, alg = prob.solver]; jac_prototype = Symbolics.jacobian_sparsity(prob), cache_eltype = eltype(prob.initial_condition), PDEkwargs...)

Solves the [`FVMProblem`](@ref) specified in `prob`. Returns a solution 
structure from `OrdinaryDiffEq.jl`. Additional keyword arguments for the solver 
can be provided with `PDEkwargs`.

A prototype for the Jacobian can be provided with `jac_prototype`.
You can control the eltype of the cache arrays for the flux vectors 
and shape coefficients using `cache_eltype`.

Note that the method used for solving the 
discretised system of ODEs is specified by `prob.solver`, or specified with `alg`.
"""
Base.@propagate_inbounds function CommonSolve.solve(prob::FVMProblem, alg=prob.solver;
    cache_eltype=eltype(prob.initial_condition), jac_prototype=float.(jacobian_sparsity(prob)),
    parallel=false, PDEkwargs...)
    ode_problem = ODEProblem(prob; jac_prototype, cache_eltype, no_saveat=:saveat ∉ keys(Dict(PDEkwargs)), parallel)#https://stackoverflow.com/questions/35157152/check-if-keyword-arguments-exist-in-julia
    sol = solve(ode_problem, alg; PDEkwargs...)
    return sol
end

"""
    ODEProblem(prob::FVMProblem; <keyword arguments>)

Method for converting the [`FVMProblem`](@ref) `prob` into an `ODEProblem`. This `ODEProblem` is 
the only returned value. You can control the eltype of the cache arrays for the flux vectors 
and shape coefficients using `cache_eltype`. If you do not need to save the solution at some specific points, 
then `no_saveat=true`; this is for the `DiscreteCallback`. The sparsity pass

# Arguments 
- `prob::FVMProblem`: The [`FVMProblem`](@ref).

# Keyword Arguments 
- `cache_eltype=eltype(prob.initial_condition)`: Element type for the cache arrays for the flux vectors and shape function coefficients.
- `jac_prototype=cache_eltype == Num ? nothing : float.(Symbolics.jacobian_sparsity(prob))`: Sparsity pattern.
- `no_saveat=true`: `true` if the solution is not being saved at some specific points; this is for the `DiscreteCallback`.
- `kwargs...`: Additional keyword arguments for the constructed `ODEProblem`.

# Output 
- `ode_problem`: The constructed `ODEProblem`.
"""
@inline function SciMLBase.ODEProblem(prob::FVMProblem;
    cache_eltype=eltype(prob.initial_condition), jac_prototype=cache_eltype ≠ eltype(prob.initial_condition) ? nothing : float.(jacobian_sparsity(prob)),
    parallel=false,
    no_saveat=true, kwargs...)
    p = FVMParameters(prob; cache_eltype, parallel)
    time_span = (prob.initial_time, prob.final_time)
    cb = dirichlet_callback(prob; no_saveat)
    if parallel
        f = ODEFunction{true}(par_fvm_eqs!; jac_prototype)
    else
        f = ODEFunction{true}(fvm_eqs!; jac_prototype)
    end
    ode_problem = ODEProblem{true}(f, prob.initial_condition, time_span, p; callback=cb, kwargs...)
    return ode_problem
end

##############################################################################
##
## INTERPOLANT
##
##############################################################################

"""
    FVMInterpolant{T}

A struct with fields `α`, `β`, `γ`, each with same type `T`, used to represent 
the linear interpolant `u(x, y) = αx + βy + γ`. Also stores the triangle and the 
nodal values. This struct is callable:

    (u::FVMInterpolant)(x, y)
    (u::FVMInterpolant)(p::Point{2, T})
    (u::FVMInterpolant)(p::AbstractVector) 

See also [`eval_interpolant`](@ref) and [`construct_mesh_interpolant`](@ref).
"""
struct FVMInterpolant{T,P}
    α::T
    β::T
    γ::T
    tr::NTuple{3,Int64}
    p₁::P
    p₂::P
    p₃::P
    u₁::T
    u₂::T
    u₃::T
    FVMInterpolant(α::T, β::T, γ::T, tr::NTuple{3,Int64}, p₁::P, p₂::P, p₃::P, u₁::T, u₂::T, u₃::T) where {T,P} = new{T,P}(α, β, γ, tr, p₁, p₂, p₃, u₁, u₂, u₃)
    FVMInterpolant{T,P}(α::T, β::T, γ::T, tr::NTuple{3,Int64}, p₁::P, p₂::P, p₃::P, u₁::T, u₂::T, u₃::T) where {T,P} = new{T,P}(α, β, γ, tr, p₁, p₂, p₃, u₁, u₂, u₃)
    FVMInterpolant(α::A, β::B, γ::C, tr::NTuple{3,Int64}, p₁::D, p₂::E, p₃::F, u₁::G, u₂::H, u₃::I) where {A,B,C,D,E,F,G,H,I} = new{promote_type(A, B, C, G, H, I),promote_type(D, E, F)}(α, β, γ, tr, p₁, p₂, p₃, u₁, u₂, u₃)
    FVMInterpolant(αβγ, tr, p, u) = new{eltype(αβγ),eltype(p)}(αβγ..., tr, p..., u...)
    FVMInterpolant(αβγ, tr, p₁, p₂, p₃, u₁, u₂, u₃) = new{eltype(αβγ),typeof(p₁)}(αβγ..., tr, p₁, p₂, p₃, u₁, u₂, u₃)
end
utype(::FVMInterpolant{T,P}) where {T,P} = T
utype(::Dict{NTuple{3,Int64},FVMInterpolant{T,P}}) where {T,P} = T
utype(::NTuple{N,Dict{NTuple{3,Int64},FVMInterpolant{T,P}}}) where {N,T,P} = T
utype(::AbstractVector{FVMInterpolant{T,P}}) where {T,P} = T

@inline (u::FVMInterpolant)(x::Real, y::Real) = u.α * x + u.β * y + u.γ
@inline (u::FVMInterpolant)(p) = u(getx(p), gety(p))

@inline nodal_values(interpolant::FVMInterpolant) = (interpolant.u₁, interpolant.u₂, interpolant.u₃)
@inline function nodal_values(interpolant::Dict{NTuple{3,Int64},FVMInterpolant{T,P}}, k) where {T,P}
    return nodal_values(interpolant[k])
end
@inline points(interpolant::FVMInterpolant) = (interpolant.p₁, interpolant.p₂, interpolant.p₃)
@inline function points(interpolant::Dict{NTuple{3,Int64},FVMInterpolant{T,P}}, k) where {T,P}
    return points(interpolant[k])
end

"""
    (u::FVMInterpolant)(x::Real, y::Real)
    (u::FVMInterpolant)(p::Point{2, T})
    (u::FVMInterpolant)(p::AbstractVector) 
    eval_interpolant(interp::FVMInterpolant, x::Real, y::Real)
    eval_interpolant(interp::FVMInterpolant, p::Point{2, T}) where {T}
    eval_interpolant(interp::Dict{Int64, FVMInterpolant{T}}, x::AbstractVector, y::AbstractVector) where {T}
    eval_interpolant(interp::Dict{Int64, FVMInterpolant{T}}, x::Real, y::Real, k::Int) where {T}
    eval_interpolant(interp::Dict{Int64, FVMInterpolant{T}}, x::AbstractVector, y::AbstractVector, k::Int) where {T}
    eval_interpolant(interp::Dict{Int64, FVMInterpolant{T}}, p) where {T}
    eval_interpolant(interp::Dict{Int64, FVMInterpolant{T}}, p, k::Int) where {T}
    eval_interpolant(interp::Dict{Int64, FVMInterpolant{T}}, p, k::Int) where {T}

Evaluate the [`FVMInterpolant`](@ref) `interp`, or the collection of, at the given points in `(x, y)` or in the point `p`. For the scalar cases,
a scalar is returned. For the vector case, a vector is returned such that `res[k]` is the value of the `k`th interpolant evaluated at the `k`th point.
"""
function eval_interpolant end
@inline function eval_interpolant(interp::FVMInterpolant, x::Real, y::Real)
    return interp(x, y)
end
@inline function eval_interpolant(interp::FVMInterpolant, p) 
    return interp(p)
end
@inline function eval_interpolant(interp::Dict{NTuple{3,Int64},FVMInterpolant{T,P}}, x::Real, y::Real, k::Int) where {T,P}
    return eval_interpolant(interp[k], x, y)
end

## Individual
@inline function construct_mesh_interpolant!(interpolants::Dict{NTuple{3,Int64},FVMInterpolant{T,P}}, shape_coeffs::AbstractVector{T}, k, mesh::FVMGeometry, u) where {T,P}
    linear_shape_function_coefficients!(shape_coeffs, u, mesh.shape_function_coeffs, k)
    interpolants[k] = FVMInterpolant(shape_coeffs, k, collect(mesh.points[:, indices(k)[1]]),collect(mesh.points[:, indices(k)[2]]),collect(mesh.points[:, indices(k)[3]]), u[indices(k)[1]], u[indices(k)[2]], u[indices(k)[3]])
    return nothing
end

## Many 
@inline function construct_mesh_interpolant!(interpolants::Dict{NTuple{3,Int64},FVMInterpolant{T,P}}, shape_coeffs::AbstractVector{T}, mesh::FVMGeometry, u) where {T,P}
    for k in mesh.elements 
        construct_mesh_interpolant!(interpolants, shape_coeffs, k, mesh, u)
    end
    return nothing
end
@inline function construct_mesh_interpolant!(interpolants::Dict{NTuple{3,Int64},FVMInterpolant{T,P}}, mesh::FVMGeometry, u) where {T,P}
    shape_coeffs = zeros(T, 3)
    construct_mesh_interpolant!(interpolants, shape_coeffs, mesh, u)
    return nothing
end
@inline function construct_mesh_interpolant(mesh::FVMGeometry, u::AbstractVector{T}) where {T}
    interpolants = Dict{NTuple{3,Int64},FVMInterpolant{T,Vector{T}}}([])
    sizehint!(interpolants, num_triangles(mesh.elements))
    construct_mesh_interpolant!(interpolants, mesh, u)
    return interpolants
end

## ODESolution 
@inline function construct_mesh_interpolant!(interpolants::Dict{NTuple{3,Int64},FVMInterpolant{T,P}}, u, mesh::FVMGeometry, sol::SciMLBase.AbstractODESolution, t::Real) where {T,P}
    copyto!(u, sol(t))
    construct_mesh_interpolant!(interpolants, mesh, u)
    return nothing
end
@inline function construct_mesh_interpolant(mesh::FVMGeometry, sol::SciMLBase.AbstractODESolution, t::Real)
    T = eltype(sol.prob.u0)
    interpolants = Dict{NTuple{3,Int64},FVMInterpolant{T,Vector{T}}}([])
    sizehint!(interpolants, num_triangles(mesh.elements))
    construct_mesh_interpolant!(interpolants, mesh, sol, t)
    return interpolants
end

## ODESolution for many times 
@inline function construct_mesh_interpolant!(interpolants, u, mesh::FVMGeometry, sol::SciMLBase.AbstractODESolution, t::AbstractVector) 
    for (i, τ) in pairs(t)
        construct_mesh_interpolant!(interpolants[i], u, mesh, sol, τ)
    end
    return nothing
end
@inline function construct_mesh_interpolant(mesh::FVMGeometry, sol::SciMLBase.AbstractODESolution, t::AbstractVector)
    m = length(t)
    interpolants = [Dict{NTuple{3,Int64},FVMInterpolant{eltype(sol.prob.u0),Vector{eltype(sol.prob.u0)}}}([]) for _ in 1:m]
    construct_mesh_interpolant!(interpolants, mesh, sol, t)
    return NTuple{m,Dict{NTuple{3,Int64},FVMInterpolant{eltype(sol.prob.u0),Vector{eltype(sol.prob.u0)}}}}(interpolants)
end

## ODESolution for all times 
@inline function construct_mesh_interpolant!(interpolants, mesh::FVMGeometry, sol::SciMLBase.AbstractODESolution)
    for (i, u) in zip(eachindex(sol), sol.u)
        construct_mesh_interpolant!(interpolants[i], mesh, u)
    end
    return nothing
end
@inline function construct_mesh_interpolant(mesh::FVMGeometry, sol::SciMLBase.AbstractODESolution)
    m = length(sol)
    interpolants = [Dict{NTuple{3,Int64},FVMInterpolant{eltype(sol.prob.u0),Vector{eltype(sol.prob.u0)}}}([]) for _ in 1:m]
    construct_mesh_interpolant!(interpolants, mesh, sol)
    return NTuple{m,Dict{NTuple{3,Int64},FVMInterpolant{eltype(sol.prob.u0),Vector{eltype(sol.prob.u0)}}}}(interpolants)
end

## General method
@inline function construct_mesh_interpolant!(interpolants, mesh::FVMGeometry, sol::SciMLBase.AbstractODESolution, t)
    u = similar(sol.prob.u0)
    construct_mesh_interpolant!(interpolants, u, mesh, sol, t)
    return nothing
end