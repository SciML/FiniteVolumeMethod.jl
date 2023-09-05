"""
    ConditionType 

This is an `Enum`-type, with three instances:

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

# functions a(x, y, t, u, p)
function get_dual_arg_types(::Type{T}, ::Type{U}, ::Type{P}) where {T,U,P}
    dU = DiffEqBase.dualgen(U)
    dT = DiffEqBase.dualgen(T)
    arg1 = Tuple{T,T,T,U,P}
    arg2 = Tuple{T,T,T,dU,P}
    arg3 = Tuple{T,T,dT,U,P}
    arg4 = Tuple{T,T,dT,dU,P}
    return (arg1, arg2, arg3, arg4)
end
function get_dual_ret_types(::Type{T}, ::Type{U}) where {T,U}
    dU = DiffEqBase.dualgen(U)
    dT = DiffEqBase.dualgen(T)
    dUT = DiffEqBase.dualgen(promote_type(T, U))
    return (U, dU, dT, dUT)
end
function wrap_functions(functions, parameters, u_type::Type{U}=Float64) where {U}
    T = Float64 # float_type
    all_arg_types = ntuple(i -> get_dual_arg_types(T, U, typeof(parameters[i])), Val(length(parameters)))
    all_ret_types = ntuple(i -> get_dual_ret_types(T, U), Val(length(parameters)))
    wrapped_functions = ntuple(i -> FunctionWrappersWrapper(functions[i], all_arg_types[i], all_ret_types[i]), Val(length(parameters)))
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
[`ConditionType`](@ref) for possible conditions - should be one of [`Neumann`](@ref), [`Dudt`](@ref), [`Dirichlet`(@ref), or [`Constrained`](@ref).

# Keyword Arguments
- `parameters=ntuple(_ -> nothing, length(functions))`

The parameters for the functions, with `parameters[i]` giving the argument `p` in `functions[i]`.
- `u_type=Float64`

The number type used for the solution.

# Outputs
The returned value is the corresponding [`BoundaryConditions`](@ref) struct.
"""
struct BoundaryConditions{F<:Tuple,P<:Tuple,C<:Tuple,UF<:Tuple}
    functions::F
    parameters::P
    condition_types::C
    unwrapped_functions::UF # not public
    function BoundaryConditions(functions::F, parameters::P, condition_types::C, unwrapped_functions::UF) where {F,P,C,UF}
        @assert all(t -> t isa ConditionType, condition_types) "The types must be instances of ConditionType."
        @assert length(functions) == length(condition_types) == length(parameters) "The number of functions, types, and parameters must be the same."
        return new{F,P,C,UF}(functions, parameters, condition_types, unwrapped_functions)
    end
end
function Base.show(io::IO, ::MIME"text/plain", bc::BoundaryConditions)
    n = length(bc.functions)
    print(io, "BoundaryConditions with $(n) boundary conditions with types $(bc.condition_types)")
end

"""
    InternalConditions(functions::Tuple=();
        dirichlet_nodes::Dict{Int,Int}=Dict{Int,Int}(),
        dudt_nodes::Dict{Int,Int}=Dict{Int,Int}();
        parameters::Tuple=ntuple(_ -> nothing, length(functions)),
        u_type=Float64)

This is a constructor for the [`InternalConditions`](@ref) struct, which holds the internal conditions for the PDE.
See also [`Conditions`](@ref) (which [`FVMProblem`](@ref) wraps this into), [`ConditionType`](@ref), and [`BoundaryConditions`](@ref).

# Arguments
- `functions::Tuple`

The functions that define the internal conditions. These are the functions refereed to in `edge_conditions` and `point_conditions`.

# Keyword Arguments
- `dirichlet_nodes::Dict{Int,Int}=Dict{Int,Int}()`

A `Dict` that stores all [`Dirichlet`](@ref) points `u` as keys, with keys mapping to indices
`idx` that refer to the corresponding condition function and parameters in `functions` and `parameters`.
- `dudt_nodes::Dict{Int,Int}=Dict{Int,Int}()`

A `Dict` that stores all [`Dudt`](@ref) points `u` as keys, with keys mapping to indices
`idx` that refer to the corresponding condition function and parameters in `functions` and `parameters`.
- `parameters::Tuple=ntuple(_ -> nothing, length(functions))`

The parameters for the functions, with `parameters[i]` giving the argument `p` in `functions[i]`.
- `u_type=Float64`

The number type used for the solution.

# Outputs
The returned value is the corresponding [`InternalConditions`](@ref) struct.

!!! note 

    When the internal conditions get merged with the boundary conditions, 
    any internal conditions that are placed onto the boundary will 
    be replaced with the boundary condition at that point on the boundary.
"""
struct InternalConditions{F<:Tuple,P<:Tuple,UF<:Tuple}
    dirichlet_nodes::Dict{Int,Int}
    dudt_nodes::Dict{Int,Int}
    functions::F
    parameters::P
    unwrapped_functions::UF # not public
    function InternalConditions(dirichlet_conditions, dudt_conditions, functions::F, parameters::P, unwrapped_functions::UF) where {F,P,UF}
        @assert length(functions) == length(parameters) "The number of functions and parameters must be the same."
        return new{F,P,UF}(dirichlet_conditions, dudt_conditions, functions, parameters, unwrapped_functions)
    end
end
function Base.show(io::IO, ::MIME"text/plain", ic::InternalConditions)
    nd = length(ic.dirichlet_nodes)
    ndt = length(ic.dudt_nodes)
    print(io, "InternalConditions with $(nd) Dirichlet nodes and $(ndt) Dudt nodes")
end

function BoundaryConditions(mesh::FVMGeometry, functions::Tuple, types::Tuple;
    parameters::Tuple=ntuple(_ -> nothing, length(functions)),
    u_type=Float64)
    nbnd_idx = DelaunayTriangulation.num_ghost_vertices(mesh.triangulation_statistics)
    @assert length(functions) == nbnd_idx "The number of boundary conditions must be the same as the number of parts of the mesh's boundary."
    wrapped_functions = wrap_functions(functions, parameters, u_type)
    return BoundaryConditions(wrapped_functions, parameters, types, functions)
end

function InternalConditions(functions::Tuple=();
    dirichlet_nodes::Dict{Int,Int}=Dict{Int,Int}(),
    dudt_nodes::Dict{Int,Int}=Dict{Int,Int}(),
    parameters::Tuple=ntuple(_ -> nothing, length(functions)),
    u_type=Float64)
    wrapped_functions = wrap_functions(functions, parameters, u_type)
    return InternalConditions(dirichlet_nodes, dudt_nodes, wrapped_functions, parameters, functions)
end

"""
    Conditions{F<:Tuple,P<:Tuple}

This is a `struct` that holds the boundary and internal conditions for the PDE.
This is not public API - the relevant public API is [`BoundaryConditions`](@ref), 
[`InternalConditions`](@ref), and [`ConditionType`](@ref).

# Fields 
- `neumann_conditions::Dict{NTuple{2,Int},Int}`

A `Dict` that stores all [`Neumann`](@ref) edges `(u, v)` as keys, with keys mapping to indices 
`idx` that refer to the corresponding condition function and parameters in `functions` and `parameters`.
- `constrained_conditions::Dict{NTuple{2,Int},Int}`

A `Dict` that stores all [`Constrained`](@ref) edges `(u, v)` as keys, with keys mapping to indices
`idx` that refer to the corresponding condition function and parameters in `functions` and `parameters`.
- `dirichlet_conditions::Dict{Int,Int}`

A `Dict` that stores all [`Dirichlet`](@ref) points `u` as keys, with keys mapping to indices
`idx` that refer to the corresponding condition function and parameters in `functions` and `parameters`.
- `dudt_conditions::Dict{Int,Int}`

A `Dict` that stores all [`Dudt`](@ref) points `u` as keys, with keys mapping to indices
`idx` that refer to the corresponding condition function and parameters in `functions` and `parameters`.
- `functions::F`

A `Tuple` of functions that correspond to the condition functions.
- `parameters::P`

A `Tuple` of parameters that correspond to the condition functions, where `parameters[i]` 
corresponds to `functions[i]`.
"""
struct Conditions{F<:Tuple,P<:Tuple,UF<:Tuple}
    neumann_edges::Dict{NTuple{2,Int},Int}
    constrained_edges::Dict{NTuple{2,Int},Int}
    dirichlet_nodes::Dict{Int,Int}
    dudt_nodes::Dict{Int,Int}
    functions::F
    parameters::P
    unwrapped_functions::UF
end
function Base.show(io::IO, ::MIME"text/plain", conds::Conditions)
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

function _rewrap_conditions(conds::Conditions, u_type::Type{U}, neqs::Val{N}) where {U,N}
    T = Float64 # float_type
    all_arg_types = ntuple(i -> get_dual_arg_types(T, N > 0 ? NTuple{N,U} : U, typeof(conds.parameters[i])), length(conds.parameters))
    all_ret_types = ntuple(i -> get_dual_ret_types(U, T), length(conds.parameters))
    wrapped_functions = ntuple(i -> FunctionWrappersWrapper(conds.unwrapped_functions[i], all_arg_types[i], all_ret_types[i]), length(conds.parameters))
    return Conditions(conds.edge_conditions, conds.point_conditions, wrapped_functions, conds.parameters, conds.unwrapped_functions)
end

function prepare_conditions(mesh::FVMGeometry, bc::BoundaryConditions, ic::InternalConditions)
    bc_functions = bc.functions
    bc_parameters = bc.parameters
    ic_functions = ic.functions
    ic_parameters = ic.parameters
    neumann_edges = Dict{NTuple{2,Int},Int}()
    constrained_edges = Dict{NTuple{2,Int},Int}()
    dirichlet_nodes = copy(ic.dirichlet_nodes)
    dudt_nodes = copy(ic.dudt_nodes)
    ne = DelaunayTriangulation.num_constrained_edges(mesh.triangulation_statistics)
    nv = DelaunayTriangulation.num_solid_vertices(mesh.triangulation_statistics)
    sizehint!(neumann_edges, ne)
    sizehint!(constrained_edges, ne)
    sizehint!(dirichlet_nodes, nv)
    sizehint!(dudt_nodes, nv)
    functions = (ic_functions..., bc_functions...)
    parameters = (ic_parameters..., bc_parameters...)
    unwrapped_functions = (ic.unwrapped_functions..., bc.unwrapped_functions...)
    conditions = Conditions(neumann_edges, constrained_edges, dirichlet_nodes, dudt_nodes, functions, parameters, unwrapped_functions)
    return conditions
end

function merge_conditions!(conditions::Conditions, mesh::FVMGeometry, bc_conditions, nif)
    tri = mesh.triangulation
    has_ghost = DelaunayTriangulation.has_ghost_triangles(tri)
    hasbnd = DelaunayTriangulation.has_boundary_nodes(tri)
    has_ghost || add_ghost_triangles!(tri)
    hasbnd || lock_convex_hull!(tri)
    bn_map = get_boundary_map(tri)
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


function Conditions(mesh::FVMGeometry, bc::BoundaryConditions, ic::InternalConditions=InternalConditions())
    conditions = prepare_conditions(mesh, bc, ic)
    merge_conditions!(conditions, mesh, bc.condition_types, length(ic.functions))
    return conditions
end