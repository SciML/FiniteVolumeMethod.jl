"""
    ConditionType 

This is an `Enum`-type, with three instances:

- [`Neumann`](@ref)
- [`Dudt`](@ref)
- [`Dirichlet`](@ref)

This is used for declaring conditions in the PDEs. See 
the associated docstrings, and also [`BoundaryConditions`](@ref)
and [`InternalConditions`](@ref).
"""
@enum ConditionType begin
    Neumann
    Dudt
    Dirichlet
end

@doc raw"""
    Neumann 

Instance of a [`ConditionType`](@ref) used for declaring that an edge 
has a `Neumann` condition. `Neumann` conditions 
take the form 

```math
\vb q(x, y, t) \vdot \vu n_\sigma = a(x, y, t, u)
```

where $\vb q$ is the flux function and $\vu n_\sigma$ is the 
outward unit normal vector field on the associated edge. 

When providing a `Neumann` condition, the function you provide 
takes the form

    a(x, y, t, u, p)

where `(x, y)` is the point, `t` is the current time, and `u` is the 
solution at the point `(x, y)` at time `t`, as above, with an extra 
argument `p` for additional parameters. 
""" Neumann

@doc raw"""
    Dudt 

Instance of a [`ConditionType`](@ref) used for declaring that an edge, 
or a point, has a `Dudt`-type boundary condition. `Dudt`-type 
conditions take the form

```math
\dv{u(x, y, t)}{t} = a(x, y, t, u).
```

When providing a `Dudt` condition, the function you provide
takes the form

    a(x, y, t, u, p)

where `(x, y)` is the point, `t` is the current time, and `u` is the
solution at the point `(x, y)` at time `t`, as above, with an extra 
argument `p` for additional parameters.
""" Dudt

@doc raw"""
    Dirichlet

Instance of a [`ConditionType`](@ref) used for declaring that an edge,
or a point, has a `Dirichlet` boundary condition. `Dirichlet`
conditions take the form

```math
u(x, y, t) = a(x, y, t, u).
```

When providing a `Dirichlet` condition, the function you provide
takes the form

    a(x, y, t, u, p)

where `(x, y)` is the point, `t` is the current time, and `u` is the
solution at the point `(x, y)` at time `t`, as above, with an extra 
argument `p` for additional parameters.
""" Dirichlet

get_dual_arg_types(::Type{T}, ::Type{U}, ::Type{P}) where {T,U,P} = (
    Tuple{T,T,T,U,P},                       # Typical signature 
    Tuple{T,T,T,DiffEqBase.dualgen(U),P},              # Signature with "u" a Dual 
    Tuple{T,T,DiffEqBase.dualgen(T),U,P},              # Signature with "t" a Dual 
    Tuple{T,T,DiffEqBase.dualgen(T),DiffEqBase.dualgen(U),P})     # Signature with "u" and "t" Duals
get_dual_ret_types(::Type{U}, ::Type{T}) where {U,T} = (U, DiffEqBase.dualgen(U), DiffEqBase.dualgen(T), DiffEqBase.dualgen(promote_type(U, T)))
function wrap_functions(functions, parameters, float_type::Type{T}=Float64, u_type::Type{U}=Float64) where {T,U}
    all_arg_types = ntuple(i -> get_dual_arg_types(T, U, typeof(parameters[i])), length(parameters))
    all_ret_types = ntuple(i -> get_dual_ret_types(U, T), length(parameters))
    wrapped_functions = ntuple(i -> FunctionWrappersWrapper(functions[i], all_arg_types[i], all_ret_types[i]), length(parameters))
    return wrapped_functions
end

"""
    BoundaryConditions(mesh::FVMGeometry, functions, conditions; parameters=nothing, u_type=Float64, float_type=Float64)

This is a constructor for the [`BoundaryConditions`](@ref) struct, which holds the boundary conditions for the PDE. 
See also [`Conditions`](@ref) (which [`FVMProblem`](@ref) wraps this into), [`ConditionType`](@ref), and [`InternalConditions`](@ref).

# Arguments
- `mesh::FVMGeometry`

The mesh on which the PDE is defined.
- `functions::Tuple`

The functions that define the boundary conditions. The `i`th function should correspond to the part of the boundary of 
the `mesh` corresponding to the `i`th boundary index, as defined in DelaunayTriangulation.jl. 
- `conditions::Tuple`

The classification for the boundary condition type corresponding to each boundary index as above. See 
[`ConditionType`](@ref) for possible conditions - should be one of [`Neumann`](@ref), [`Dudt`](@ref), and [`Dirichlet`(@ref).

# Keyword Arguments
- `parameters=ntuple(_ -> nothing, length(functions))`

The parameters for the functions, with `parameters[i]` giving the argument `p` in `functions[i]`.
- `u_type=Float64`

The number type used for the solution.
- `float_type=Float64`

The number type used for representing the coordinates of points.

# Outputs
The returned value is the corresponding [`BoundaryConditions`](@ref) struct.
"""
struct BoundaryConditions{F<:Tuple,P<:Tuple,C<:Tuple}
    functions::F
    parameters::P
    condition_types::C
    function BoundaryConditions(functions::F, parameters::P, condition_types::C) where {F,P,C}
        @assert all(t -> t isa ConditionType, condition_types) "The types must be instances of ConditionType."
        @assert length(functions) == length(condition_types) == length(parameters) "The number of functions, types, and parameters must be the same."
        return new{F,P,C}(functions, parameters, condition_types)
    end
end

"""
    InternalConditions(functions::Tuple,
        point_conditions::Dict{Int,Tuple{ConditionType,Int}}=Dict{Int,Tuple{ConditionType,Int}}();
        parameters::Tuple=ntuple(_ -> nothing, length(functions)),
        u_type=Float64,
        float_type=Float64)

This is a constructor for the [`InternalConditions`](@ref) struct, which holds the internal conditions for the PDE.
See also [`Conditions`](@ref) (which [`FVMProblem`](@ref) wraps this into), [`ConditionType`](@ref), and [`BoundaryConditions`](@ref).

# Arguments
- `functions::Tuple`

The functions that define the internal conditions. These are the functions refereed to in `edge_conditions` and `point_conditions`.
- `point_conditions::Dict{Int,Tuple{ConditionType,Int}}=Dict{Int,Tuple{ConditionType,Int}}()`

A `Dict` that maps a vertex `u`, referring to a point in the associated triangulation, to a `Tuple` of the form
`(ConditionType, idx)`, where `ConditionType` is either [`Dudt`](@ref) or [`Dirichlet`](@ref) from [`ConditionType`](@ref),
and `idx` is the index of the associated condition function and parameters in `functions` and `parameters`.

# Keyword Arguments
- `parameters::Tuple=ntuple(_ -> nothing, length(functions))`

The parameters for the functions, with `parameters[i]` giving the argument `p` in `functions[i]`.
- `u_type=Float64`

The number type used for the solution.
- `float_type=Float64`

The number type used for representing the coordinates of points.

# Outputs
The returned value is the corresponding [`InternalConditions`](@ref) struct.
"""
struct InternalConditions{F,P}
    point_conditions::Dict{Int,Tuple{ConditionType,Int}}
    functions::F
    parameters::P
    function InternalConditions(point_conditions, functions::F, parameters::P) where {F,P}
        @assert length(functions) == length(parameters) "The number of functions and parameters must be the same."
        return new{F,P}(point_conditions, functions, parameters)
    end
end

function BoundaryConditions(mesh::FVMGeometry, functions::Tuple, types::Tuple;
    parameters::Tuple=ntuple(_ -> nothing, length(functions)),
    u_type=Float64,
    float_type=Float64)
    nbnd_idx = DelaunayTriangulation.num_ghost_vertices(mesh.triangulation_statistics)
    @assert length(functions) == nbnd_idx "The number of boundary conditions must be the same as the number of parts of the mesh's boundary."
    wrapped_functions = wrap_functions(functions, parameters, u_type, float_type)
    return BoundaryConditions(wrapped_functions, parameters, types)
end

function InternalConditions(functions::Tuple=(), point_conditions::Dict{Int,Tuple{ConditionType,Int}}=Dict{Int,Tuple{ConditionType,Int}}();
    parameters::Tuple=ntuple(_ -> nothing, length(functions)),
    u_type=Float64,
    float_type=Float64)
    wrapped_functions = wrap_functions(functions, parameters, u_type, float_type)
    return InternalConditions(point_conditions, wrapped_functions, parameters)
end

"""
    Conditions{F<:Tuple,P<:Tuple}

This is a `struct` that holds the boundary and internal conditions for the PDE.

See also [`BoundaryConditions`](@ref), [`InternalConditions`](@ref),
and [`ConditionType`](@ref).

# Fields 
- `neumann_conditions::Dict{NTuple{2,Int},Int}`

A `Dict` that maps an oriented edge `(u, v)`, with vertices referring to points in the associated triangulation, 
to the index `idx` of the associated condition function and parameters in `functions` and `parameters`. The enforced 
condition on these edges are [`Neumann`](@ref) conditions.
- `point_conditions::Dict{Int,Tuple{ConditionType,Int}}`

A `Dict` that maps a vertex `u`, referring to a point in the associated triangulation, to a `Tuple` of the form 
`(ConditionType, idx)`, where `ConditionType` is either [`Dudt`](@ref) or [`Dirichlet`](@ref) from [`ConditionType`](@ref),
and `idx` is the index of the associated condition function and parameters in `functions` and `parameters`.
- `functions::F`

A `Tuple` of functions that correspond to the conditions in `neumann_conditions` and `point_conditions`.
- `parameters::P`

A `Tuple` of parameters that correspond to the conditions in `neumann_conditions` and `point_conditions`, where `parameters[i]` 
corresponds to `functions[i]`.
"""
struct Conditions{F<:Tuple,P<:Tuple}
    edge_conditions::Dict{NTuple{2,Int},Int}
    point_conditions::Dict{Int,Tuple{ConditionType,Int}}
    functions::F
    parameters::P
end

function prepare_conditions(mesh::FVMGeometry, bc::BoundaryConditions, ic::InternalConditions)
    bc_functions = bc.functions
    bc_parameters = bc.parameters
    ic_functions = ic.functions
    ic_parameters = ic.parameters
    edge_conditions = Dict{NTuple{2,Int},Int}()
    point_conditions = copy(ic.point_conditions)
    ne = DelaunayTriangulation.num_constrained_edges(mesh.triangulation_statistics)
    nv = DelaunayTriangulation.num_solid_vertices(mesh.triangulation_statistics)
    sizehint!(edge_conditions, ne)
    sizehint!(point_conditions, nv)
    functions = (ic_functions..., bc_functions...)
    parameters = (ic_parameters..., bc_parameters...)
    conditions = Conditions(ic_edge_conditions, ic_point_conditions, functions, parameters)
    return conditions, bc.condition_types
end

# (cur, prev) ↦ (merged, b), where b is `true` if replaced with the previous condition, and `false` otherwise.
function reconcile_conditions(cur_condition::ConditionType, prev_condition::ConditionType)
    cur_condition == Dirichlet && return (Dirichlet, false)
    if cur_condition == Dudt
        prev_condition == Dirichlet && return (Dirichlet, true)
        return (Dudt, false)
    else # Neumann  
        prev_condition == Dirichlet && return (Dirichlet, true)
        prev_condition == Dudt && return (Dudt, true)
        return (Neumann, false)
    end
end

# returns (merged_condition, updated_bc_number)
function get_edge_condition(bc_conditions, tri, i, j, bc_number, nif)
    prev_boundary_index = get_adjacent(tri, j, i)
    cur_condition = bc_conditions[bc_number]
    prev_condition = bc_conditions[-prev_boundary_index]
    merged_condition, replaced = reconcile_conditions(cur_condition, prev_condition)
    return merged_condition, replaced ? -prev_boundary_index + nif : nif + bc_number
end

function add_condition!(conditions::Conditions, condition, bc_number, i, j)
    if condition == Neumann
        conditions.edge_conditions[(i, j)] = bc_number
    else
        conditions.point_conditions[i] = (condition, bc_number)
    end
    return nothing
end

function merge_conditions!(conditions::Conditions, mesh::FVMGeometry, bc_conditions, nif)
    tri = mesh.triangulation
    has_ghost = DelaunayTriangulation.has_ghost_triangles(tri)
    hasbnd = DelaunayTriangulation.has_boundary_nodes(tri)
    has_ghost || add_ghost_triangles!(tri)
    hasbnd || lock_convex_hull!(tri)
    bn_map = get_boundary_map(tri)
    for (bc_number, (boundary_index, segment_index)) in enumerate(bn_map)
        bn_nodes = get_boundary_nodes(tri, segment_index)
        nedges = num_boundary_edges(bn_nodes)
        v = get_boundary_nodes(bn_nodes, 1)
        u = DelaunayTriangulation.get_left_boundary_node(tri, v, boundary_index)
        merged_condition, updated_bc_number = get_edge_condition(bc_conditions, tri, u, v, bc_number, nif)
        w = get_boundary_nodes(bn_nodes, 2)
        add_condition!(conditions, merged_condition, updated_bc_number, v, w)
        for i in 2:nedges
            u = v
            v = get_boundary_nodes(bn_nodes, i)
            w = get_boundary_nodes(bn_nodes, i + 1)
            merged_condition, updated_bc_number = get_edge_condition(bc_conditions, tri, u, v, bc_number, nif)
            add_condition!(conditions, merged_condition, updated_bc_number, v, w)
        end
    end
    hasbnd || unlock_convex_hull!(tri)
    has_ghost || delete_ghost_triangles!(tri)
    return conditions
end

function Conditions(mesh::FVMGeometry, bc::BoundaryConditions, ic::InternalConditions=InternalConditions())
    conditions = prepare_conditions(mesh, bc, ic)
    merge_conditions!(conditions, mesh, bc.condition_types, length(ic.functions))
    return conditions
end