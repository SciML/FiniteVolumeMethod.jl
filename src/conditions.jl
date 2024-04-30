"""
    ParametrisedFunction{F<:Function,P} <: Function

This is a `struct` that wraps a function `f` and some parameters `p` into a single object.

# Fields
- `fnc::F`

The function that is wrapped.
- `parameters::P`

The parameters that are wrapped.
"""
struct ParametrisedFunction{F<:Function,P} <: Function
    fnc::F
    parameters::P
end
@inline (f::ParametrisedFunction{F,P})(args...) where {F,P} = f.fnc(args..., f.parameters)

"""
    ConditionType 

This is an `Enum`-type, with four instances:

- [`Neumann`](@ref)
- [`Dudt`](@ref)
- [`Dirichlet`](@ref)
- [`Constrained`](@ref)

This is used for declaring conditions in the PDEs. See 
the associated docstrings, and also [`BoundaryConditions`](@ref)
and [`InternalConditions`](@ref).
"""
@enum ConditionType begin
    Neumann
    Dudt
    Dirichlet
    Constrained
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
outward unit normal vector field on the associated edge (meaning, for example, 
the normal vector to an edge `pq` would point to the right of `pq`).

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

@doc raw"""
    Constrained 

Instance of a [`ConditionType`](@ref) used for declaring that an edge
has a `Constrained` condition. When an edge has this condition associated with it, 
it will be treated as any normal edge and no boundary condition will be applied.
With this condition, it is assumed that you will later setup your problem as a 
differential-algebraic equation (DAE) and provide the appropriate constraints.
See the docs for some examples.

When you provide a `Constrained` condition, for certain technical reasons 
you do still need to provide a function that corresponds to it in the function list 
provided to [`BoundaryConditions`](@ref). For this function, any function will work, 
e.g. `sin` - it will not be called. The proper constraint function is to be provided after-the-fact, 
as explained in the docs.
""" Constrained

@inline function wrap_functions(functions::Tuple, parameters)
    wrapped_functions = ntuple(i -> ParametrisedFunction(functions[i], parameters[i]), Val(length(parameters)))
    return wrapped_functions
end

"""
    BoundaryConditions(mesh::FVMGeometry, functions, conditions; parameters=nothing)

This is a constructor for the [`BoundaryConditions`](@ref) struct, which holds the boundary conditions for the PDE. 
See also [`Conditions`](@ref) (which [`FVMProblem`](@ref) wraps this into), [`ConditionType`](@ref), and [`InternalConditions`](@ref).

# Arguments
- `mesh::FVMGeometry`

The mesh on which the PDE is defined.
- `functions::Union{<:Tuple,<:Function}`

The functions that define the boundary conditions. The `i`th function should correspond to the part of the boundary of 
the `mesh` corresponding to the `i`th boundary index, as defined in DelaunayTriangulation.jl. 
- `conditions::Union{<:Tuple,<:ConditionType}`

The classification for the boundary condition type corresponding to each boundary index as above. See 
[`ConditionType`](@ref) for possible conditions - should be one of [`Neumann`](@ref), [`Dudt`](@ref), [`Dirichlet`](@ref), or [`Constrained`](@ref).

# Keyword Arguments
- `parameters=ntuple(_ -> nothing, length(functions))`

The parameters for the functions, with `parameters[i]` giving the argument `p` in `functions[i]`.

# Outputs
The returned value is the corresponding [`BoundaryConditions`](@ref) struct.
"""
struct BoundaryConditions{F<:Tuple,C<:Tuple}
    functions::F
    condition_types::C
    function BoundaryConditions(functions::F, condition_types::C) where {F,C}
        @assert length(functions) == length(condition_types) "The number of functions and types must be the same."
        @assert all(t -> t isa ConditionType, condition_types) "The condition types must be ConditionType instances."
        return new{F,C}(functions, condition_types)
    end
end
function Base.show(io::IO, ::MIME"text/plain", bc::BoundaryConditions)
    n = length(bc.functions)
    if n > 1
        print(io, "BoundaryConditions with $(n) boundary conditions with types $(bc.condition_types)")
    else
        print(io, "BoundaryConditions with $(n) boundary condition with type $(bc.condition_types[1])")
    end
end

"""
    InternalConditions(functions=();
        dirichlet_nodes::Dict{Int,Int}=Dict{Int,Int}(),
        dudt_nodes::Dict{Int,Int}=Dict{Int,Int}(),
        parameters::Tuple=ntuple(_ -> nothing, length(functions)))

This is a constructor for the [`InternalConditions`](@ref) struct, which holds the internal conditions for the PDE.
See also [`Conditions`](@ref) (which [`FVMProblem`](@ref) wraps this into), [`ConditionType`](@ref), and [`BoundaryConditions`](@ref).

# Arguments
- `functions::Union{<:Tuple,<:Function}=()`

The functions that define the internal conditions. These are the functions referred to in `edge_conditions` and `point_conditions`.

# Keyword Arguments
- `dirichlet_nodes::Dict{Int,Int}=Dict{Int,Int}()`

A `Dict` that stores all [`Dirichlet`](@ref) points `u` as keys, with keys mapping to indices
`idx` that refer to the corresponding condition function and parameters in `functions` and `parameters`.
- `dudt_nodes::Dict{Int,Int}=Dict{Int,Int}()`

A `Dict` that stores all [`Dudt`](@ref) points `u` as keys, with keys mapping to indices
`idx` that refer to the corresponding condition function and parameters in `functions` and `parameters`.
- `parameters::Tuple=ntuple(_ -> nothing, length(functions))`

The parameters for the functions, with `parameters[i]` giving the argument `p` in `functions[i]`.

# Outputs
The returned value is the corresponding [`InternalConditions`](@ref) struct.

!!! note 

    When the internal conditions get merged with the boundary conditions, 
    any internal conditions that are placed onto the boundary will 
    be replaced with the boundary condition at that point on the boundary.
"""
struct InternalConditions{F<:Tuple}
    dirichlet_nodes::Dict{Int,Int}
    dudt_nodes::Dict{Int,Int}
    functions::F
    function InternalConditions(dirichlet_conditions, dudt_conditions, functions::F) where {F}
        return new{F}(dirichlet_conditions, dudt_conditions, functions)
    end
end
function Base.show(io::IO, ::MIME"text/plain", ic::InternalConditions)
    nd = length(ic.dirichlet_nodes)
    ndt = length(ic.dudt_nodes)
    print(io, "InternalConditions with $(nd) Dirichlet nodes and $(ndt) Dudt nodes")
end

function BoundaryConditions(mesh::FVMGeometry, functions::Tuple, types::Tuple;
    parameters::Tuple=ntuple(_ -> nothing, length(functions)))
    nbnd_idx = DelaunayTriangulation.num_ghost_vertices(mesh.triangulation_statistics)
    @assert length(functions) == nbnd_idx "The number of boundary conditions must be the same as the number of parts of the mesh's boundary."
    wrapped_functions = wrap_functions(functions, parameters)
    return BoundaryConditions(wrapped_functions, types)
end
function BoundaryConditions(mesh::FVMGeometry, functions::Function, types::ConditionType;
    parameters=nothing)
    return BoundaryConditions(mesh, (functions,), (types,), parameters=(parameters,))
end

@inline function InternalConditions(functions::Tuple=();
    dirichlet_nodes::Dict{Int,Int}=Dict{Int,Int}(),
    dudt_nodes::Dict{Int,Int}=Dict{Int,Int}(),
    parameters::Tuple=ntuple(_ -> nothing, length(functions)))
    wrapped_functions = wrap_functions(functions, parameters)
    return InternalConditions(dirichlet_nodes, dudt_nodes, wrapped_functions)
end
@inline function InternalConditions(functions::Function;
    dirichlet_nodes::Dict{Int,Int}=Dict{Int,Int}(),
    dudt_nodes::Dict{Int,Int}=Dict{Int,Int}(),
    parameters=nothing)
    return InternalConditions((functions,); dirichlet_nodes, dudt_nodes, parameters=(parameters,))
end

abstract type AbstractConditions end

"""
    Conditions{F} <: AbstractConditions

This is a `struct` that holds the boundary and internal conditions for the PDE. The constructor is 

    Conditions(mesh::FVMGeometry, bc::BoundaryConditions, ic::InternalConditions=InternalConditions())

The fields are:

# Fields 
- `neumann_conditions::Dict{NTuple{2,Int},Int}`

A `Dict` that stores all [`Neumann`](@ref) edges `(u, v)` as keys, with keys mapping to indices 
`idx` that refer to the corresponding condition function and parameters in `functions`.
- `constrained_conditions::Dict{NTuple{2,Int},Int}`

A `Dict` that stores all [`Constrained`](@ref) edges `(u, v)` as keys, with keys mapping to indices
`idx` that refer to the corresponding condition function and parameters in `functions`.
- `dirichlet_conditions::Dict{Int,Int}`

A `Dict` that stores all [`Dirichlet`](@ref) points `u` as keys, with keys mapping to indices
`idx` that refer to the corresponding condition function and parameters in `functions`.
- `dudt_conditions::Dict{Int,Int}`

A `Dict` that stores all [`Dudt`](@ref) points `u` as keys, with keys mapping to indices
`idx` that refer to the corresponding condition function and parameters in `functions`.
- `functions::F<:Tuple`

The functions that define the boundary and internal conditions. The `i`th function should correspond to the part of the boundary of
the `mesh` corresponding to the `i`th boundary index, as defined in DelaunayTriangulation.jl. The `i`th function is stored 
as a [`ParametrisedFunction`](@ref).
"""
struct Conditions{F<:Tuple} <: AbstractConditions
    neumann_edges::Dict{NTuple{2,Int},Int}
    constrained_edges::Dict{NTuple{2,Int},Int}
    dirichlet_nodes::Dict{Int,Int}
    dudt_nodes::Dict{Int,Int}
    functions::F
    @inline function Conditions(neumann_edges, constrained_edges, dirichlet_nodes, dudt_nodes, functions::F) where {F}
        return new{F}(neumann_edges, constrained_edges, dirichlet_nodes, dudt_nodes, functions)
    end
end
function Base.show(io::IO, ::MIME"text/plain", conds::AbstractConditions)
    nn = length(conds.neumann_edges)
    nc = length(conds.constrained_edges)
    nd = length(conds.dirichlet_nodes)
    ndt = length(conds.dudt_nodes)
    println(io, "Conditions with")
    println(io, "   $(nn) Neumann edges")
    println(io, "   $(nc) Constrained edges")
    println(io, "   $(nd) Dirichlet nodes")
    print(io, "   $(ndt) Dudt nodes")
end

"""
    get_dudt_fidx(conds, node)

Get the index of the function that corresponds to the [`Dudt`](@ref) condition at `node`.
"""
@inline get_dudt_fidx(conds::AbstractConditions, node) = conds.dudt_nodes[node]

"""
    get_neumann_fidx(conds, i, j)

Get the index of the function that corresponds to the [`Neumann`](@ref) condition at the edge `(i, j)`.
"""
@inline get_neumann_fidx(conds::AbstractConditions, i, j) = conds.neumann_edges[(i, j)]

"""
    get_dirichlet_fidx(conds, node)

Get the index of the function that corresponds to the [`Dirichlet`](@ref) condition at `node`.
"""
@inline get_dirichlet_fidx(conds::AbstractConditions, node) = conds.dirichlet_nodes[node]

"""
    get_constrained_fidx(conds, i, j)

Get the index of the function that corresponds to the [`Constrained`](@ref) condition at the edge `(i, j)`.
"""
@inline get_constrained_fidx(conds::AbstractConditions, i, j) = conds.constrained_edges[(i, j)]

@inline get_f(conds::Conditions{F}, fidx) where {F} = conds.functions[fidx]

"""
    eval_condition_fnc(conds, fidx, x, y, t, u)

Evaluate the function that corresponds to the condition at `fidx` at the point `(x, y)` at time `t` with solution `u`.
"""
@inline eval_condition_fnc(conds::Conditions, fidx, x, y, t, u) = eval_fnc_in_het_tuple(conds.functions, fidx, x, y, t, u)

"""
    is_dudt_node(conds, node)

Check if `node` has a [`Dudt`](@ref) condition.
"""
@inline is_dudt_node(conds::AbstractConditions, node) = haskey(conds.dudt_nodes, node)
"""
    is_neumann_edge(conds, i, j)

Check if the edge `(i, j)` has a [`Neumann`](@ref) condition.
"""
@inline is_neumann_edge(conds::AbstractConditions, i, j) = haskey(conds.neumann_edges, (i, j))

"""
    is_dirichlet_node(conds, node)

Check if `node` has a [`Dirichlet`](@ref) condition.
"""
@inline is_dirichlet_node(conds::AbstractConditions, node) = haskey(conds.dirichlet_nodes, node)

"""
    is_constrained_edge(conds, i, j)

Check if the edge `(i, j)` has a [`Constrained`](@ref) condition.
"""
@inline is_constrained_edge(conds::AbstractConditions, i, j) =  haskey(conds.constrained_edges, (i, j))

"""
    has_constrained_edges(conds)

Check if any edge has a [`Constrained`](@ref) condition.
"""
@inline has_constrained_edges(conds::AbstractConditions) = !isempty(conds.constrained_edges)

"""
    has_neumann_edges(conds)

Check if any edge has a [`Neumann`](@ref) condition.
"""
@inline has_neumann_edges(conds::AbstractConditions) = !isempty(conds.neumann_edges)

"""
    has_condition(conds, node)

Check if `node` has any condition.
"""
@inline has_condition(conds::AbstractConditions, node) = is_dudt_node(conds, node) || is_dirichlet_node(conds, node)

"""
    has_dirichlet_nodes(conds)

Check if any node has a [`Dirichlet`](@ref) condition.
"""
@inline has_dirichlet_nodes(conds::AbstractConditions) = !isempty(conds.dirichlet_nodes)

"""
    get_dirichlet_nodes(conds)

Get all nodes that have a [`Dirichlet`](@ref) condition.
"""
@inline get_dirichlet_nodes(conds::AbstractConditions) = conds.dirichlet_nodes

"""
    has_dudt_nodes(conds)

Check if any node has a [`Dudt`](@ref) condition.
"""
@inline has_dudt_nodes(conds::AbstractConditions) = !isempty(conds.dudt_nodes)

"""
    get_dudt_nodes(conds)

Get all nodes that have a [`Dudt`](@ref) condition.
"""
@inline get_dudt_nodes(conds::AbstractConditions) = conds.dudt_nodes

"""
    get_neumann_edges(conds)

Get all edges that have a [`Neumann`](@ref) condition.
"""
@inline get_neumann_edges(conds::AbstractConditions) = conds.neumann_edges

"""
    get_constrained_edges(conds)

Get all edges that have a [`Constrained`](@ref) condition.
"""
@inline get_constrained_edges(conds::AbstractConditions) = conds.constrained_edges

@inline function prepare_conditions(mesh::FVMGeometry, bc::BoundaryConditions, ic::InternalConditions)
    bc_functions = bc.functions
    ic_functions = ic.functions
    neumann_edges = Dict{NTuple{2,Int},Int}()
    constrained_edges = Dict{NTuple{2,Int},Int}()
    dirichlet_nodes = copy(ic.dirichlet_nodes)
    dudt_nodes = copy(ic.dudt_nodes)
    ne = DelaunayTriangulation.num_segments(mesh.triangulation_statistics)
    nv = DelaunayTriangulation.num_solid_vertices(mesh.triangulation_statistics)
    sizehint!(neumann_edges, ne)
    sizehint!(constrained_edges, ne)
    sizehint!(dirichlet_nodes, nv)
    sizehint!(dudt_nodes, nv)
    functions = (ic_functions..., bc_functions...)
    conditions = Conditions(neumann_edges, constrained_edges, dirichlet_nodes, dudt_nodes, functions)
    return conditions
end

function merge_conditions!(conditions::Conditions, mesh::FVMGeometry, bc_conditions, nif)
    tri = mesh.triangulation
    has_ghost = DelaunayTriangulation.has_ghost_triangles(tri)
    hasbnd = DelaunayTriangulation.has_boundary_nodes(tri)
    has_ghost || add_ghost_triangles!(tri)
    hasbnd || lock_convex_hull!(tri)
    bn_map = get_ghost_vertex_map(tri)
    for (bc_number, (_, segment_index)) in enumerate(bn_map)
        bn_nodes = get_boundary_nodes(tri, segment_index)
        nedges = num_boundary_edges(bn_nodes)
        for i in 1:nedges
            u = get_boundary_nodes(bn_nodes, i)
            v = get_boundary_nodes(bn_nodes, i + 1)
            condition = bc_conditions[bc_number]
            updated_bc_number = bc_number + nif
            if condition == Neumann
                conditions.neumann_edges[(u, v)] = updated_bc_number
            elseif condition == Constrained
                conditions.constrained_edges[(u, v)] = updated_bc_number
            elseif condition == Dirichlet
                conditions.dirichlet_nodes[u] = updated_bc_number
                conditions.dirichlet_nodes[v] = updated_bc_number
            else # Dudt 
                # Strictly speaking, we do need to take care that no Dudt 
                # nodes are also assigned as Dirichlet nodes, since 
                # Dirichlet conditions take precedence over Dudt conditions. 
                # However, in the code we also defend against this by checking 
                # for Dirichlet first, so this check is not _technically_
                # needed at all.
                conditions.dudt_nodes[u] = updated_bc_number
                conditions.dudt_nodes[v] = updated_bc_number
            end
        end
    end
    hasbnd || unlock_convex_hull!(tri)
    has_ghost || delete_ghost_triangles!(tri)
    return conditions
end

@inline function Conditions(mesh::FVMGeometry, bc::BoundaryConditions, ic::InternalConditions=InternalConditions())
    conditions = prepare_conditions(mesh, bc, ic)
    merge_conditions!(conditions, mesh, bc.condition_types, length(ic.functions))
    return conditions
end

struct SimpleConditions <: AbstractConditions # in 2.0, this needs to be part of Conditions. need this for type stability in FVMSystems
    neumann_edges::Dict{NTuple{2,Int},Int}
    constrained_edges::Dict{NTuple{2,Int},Int}
    dirichlet_nodes::Dict{Int,Int}
    dudt_nodes::Dict{Int,Int}
end

