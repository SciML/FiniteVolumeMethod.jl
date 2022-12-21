"""
    is_dirichlet_type(type)

Returns `type ∈ (:Dirichlet, :D, :dirichlet, "Dirichlet", "D", "dirichlet")`.
"""
is_dirichlet_type(type) = type ∈ (:Dirichlet, :D, :dirichlet, "Dirichlet", "D", "dirichlet")
"""
    is_neumann_type(type)

Returns `type ∈ (:Neumann, :N, :neumann, "Neumann", "N", "neumann")`.
"""
is_neumann_type(type) = type ∈ (:Neumann, :N, :neumann, "Neumann", "N", "neumann")
"""
    is_dudt_type(type)

Returns `type ∈ (:Dudt, :dudt, "Dudt", "dudt", "du/dt")`.
"""
is_dudt_type(type) = type ∈ (:Dudt, :dudt, "Dudt", "dudt", "du/dt")

"""
    BoundaryConditions{BNV,F,P,DN,NN,DuN,INN,BMI,MBI,TM}

Information representing the boundary conditions for the PDE. 

# Fields 
- `boundary_node_vector::BNV`: The vector of vectors such that each nested vector is the list of nodes for each segment, given in counter-clockwise order, and such that `first(BNV[i]) == last(BNV[i-1])`.
- `functions::F`: The `Tuple` of boundary condition functions for each boundary segment, with `functions[i]` corresponding to the `i`th segment. These functions must take the form `f(x, y, t, u, p)`.
- `parameters::P`: The `Tuple` of arguments `p` for each boundary condition function, with `parameters[i]` corresponding to `functions[i]`.
- `dirichlet_nodes::DN`: The indices of the nodes on the boundary that are of Dirichlet type. 
- `neumann_nodes::NN`: The indices of the nodes on the boundary that are of Neumann type. 
- `dudt_nodes::DuN`: The indices of the nodes on the boundary that are of time-dependent Dirichlet type, i.e. of the form `du/dt = f(x, y, t, u, p)`.
- `interior_or_neumann_nodes`::INN`: The nodes that are either interior or neumann nodes. 
- `boundary_to_mesh_idx::BMI`: If the boundary nodes are in the order `(b₁, b₂, …, bᵢ, …)`, then this is a map that takes the index order `i` to the corresponding index in the mesh, i.e. to `bᵢ`.
- `mesh_to_boundary_idx::MBI`: The inverse map of `boundary_to_mesh_idx`.
- `type_map::TM`: Given a node, maps it to the segment number that it belongs to.
"""
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

"""
    BoundaryConditions{BNV,F,P,DN,NN,DuN,INN,BMI,MBI,TM}

Information representing the boundary conditions for the PDE. 

# Fields 
- `boundary_node_vector::BNV`

The vector of vectors such that each nested vector is the list of nodes for each segment, given in counter-clockwise order, and such that `first(BNV[i]) == last(BNV[i-1])`.
- `functions::F`

The `Tuple` of boundary condition functions for each boundary segment, with `functions[i]` corresponding to the `i`th segment. These functions must take the form `f(x, y, t, u, p)`.
- `parameters::P`

The `Tuple` of arguments `p` for each boundary condition function, with `parameters[i]` corresponding to `functions[i]`.
- `dirichlet_nodes::DN`

The indices of the nodes on the boundary that are of Dirichlet type. 
- `neumann_nodes::NN`

The indices of the nodes on the boundary that are of Neumann type. 
- `dudt_nodes::DuN`

The indices of the nodes on the boundary that are of time-dependent Dirichlet type, i.e. of the form `du/dt = f(x, y, t, u, p)`.
- `interior_or_neumann_nodes`::INN`

The nodes that are either interior or neumann nodes. 
- `boundary_to_mesh_idx::BMI`

If the boundary nodes are in the order `(b₁, b₂, …, bᵢ, …)`, then this is a map that takes the index order `i` to the corresponding index in the mesh, i.e. to `bᵢ`.
- `mesh_to_boundary_idx::MBI`

The inverse map of `boundary_to_mesh_idx`.
- `type_map::TM`

Given a node, maps it to the segment number that it belongs to.

# Constructors 

    BoundaryConditions(mesh::FVMGeometry, functions, types, boundary_node_vector;
        params=Tuple(nothing for _ in (functions isa Function ? [1] : eachindex(functions))),
        u_type=Float64, float_type=Float64)

Constructor for the [`BoundaryConditions`](@ref) struct. 

## Arguments 
- `mesh::FVMGeometry`: The [`FVMGeometry`](@ref) for the mesh. 
- `functions`: The functions for each boundary segment, taking the forms `f(x, y, t, u, p)`. Can be a single function, doesn't have to be in a container (as long as only one segment is needed).
- `types`: The classification for the boundary condition type on each segment. See [`is_dirichlet_type`](@ref), [`is_neumann_type`](@ref), and [`is_dudt_type`](@ref) for the possible values here. `types[i]` is the classification for the `i`th segment. 
- `boundary_node_vector`: The boundary node vector for the struct: The vector of vectors such that each nested vector is the list of nodes for each segment, given in counter-clockwise order, and such that `first(boundary_node_vector[i]) == last(boundary_node_vector[i-1])`.

## Keyword Arguments 
- `params=Tuple(nothing for _ in (functions isa Function ? [1] : eachindex(functions)))`: The parameters for the functions, with `params[i]` giving the argument `p` in `functions[i]`.
- `u_type=Float64`: The number type used for the solution. 
- `float_type=Float64`: The number type used for representing the coordinates of points. 

## Outputs 
The returned value is the corresponding [`BoundaryConditions`](@ref) struct. 
"""
function BoundaryConditions end 
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