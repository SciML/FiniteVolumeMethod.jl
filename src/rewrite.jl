"""
    FVMGeometry(tri::Triangulation)

This is a constructor for the [`FVMGeometry`](@ref) struct, which holds the mesh and associated data for the PDE.

It is assumed that all vertices in `tri` are in the triangulation, meaning `v` is in `tri` for each `v` in `each_point_index(tri)`.
"""
struct FVMGeometry{T,S,V,C}
    triangulation::T
    triangulation_statistics::S
    cv_volumes::Vector{Float64} # don't need to use a Dict... all vertices should be present in the mesh.
    shape_function_coefficients::Dict{NTuple{3,Int},NTuple{9,Float64}}
end
function FVMGeometry(tri::Triangulation)
    has_ghost = DelaunayTriangulation.has_ghost_triangles(tri)
    has_ghost || add_ghost_triangles!(tri)
    stats = statistics(tri)
    nn = DelaunayTriangulation.num_solid_vertices(stats)
    nt = DelaunayTriangulation.num_solid_triangles(stats)
    cv_volumes = zeros(Int, nn)
    shape_function_coefficients = Dict{NTuple{3,Int},NTuple{9,Float64}}()
    sizehint!(cv_volumes, nn)
    sizehint!(shape_function_coefficients, nt)
    for T in each_solid_triangle(tri)
        i, j, k = indices(T)
        p, q, r = get_point(tri, i, j, k)
        px, py = getxy(p)
        qx, qy = getxy(q)
        rx, ry = getxy(r)
        # Get the centroid of the triangle, and the midpoint of each edge
        centroid = DelaunayTriangulation.get_centroid(stats, T)
        m1, m2, m3 = DelaunayTriangulation.get_edge_midpoints(stats, T)
        # Now we need to connect the centroid to each vertex 
        cx, cy = getxy(centroid)
        pcx, pcy = cx - pcx, cy - pcy
        qcx, qcy = cx - qcx, cy - qcy
        rcx, rcy = cx - rcx, cy - rcy
        # Next, connect all the midpoints to each other
        m1x, m1y = getxy(m1)
        m2x, m2y = getxy(m2)
        m3x, m3y = getxy(m3)
        m13x, m13y = m1x - m3x, m1y - m3y
        m21x, m21y = m2x - m1x, m2y - m1y
        m32x, m32y = m3x - m2x, m3y - m2y
        # We can now contribute the portion of each vertex's control volume inside the triangle to its total volume 
        S₁ = 1 / 2 * abs(pcx * m13y - pcy * m13x)
        S₂ = 1 / 2 * abs(qcx * m21y - qcy * m21x)
        S₃ = 1 / 2 * abs(rcx * m32y - rcy * m32x)
        cv_volumes[i] += S₁
        cv_volumes[j] += S₂
        cv_volumes[k] += S₃
        # Lastly, we need to compute the shape function coefficients
        Δ = qx * ry - qy * rx - px * ry + rx * py + px * qy - qx * py
        s₁ = (qy - ry) / Δ
        s₂ = (ry - py) / Δ
        s₃ = (py - qy) / Δ
        s₄ = (rx - qx) / Δ
        s₅ = (px - rx) / Δ
        s₆ = (qx - px) / Δ
        s₇ = (qx * ry - rx * qy) / Δ
        s₈ = (rx * py - px * ry) / Δ
        s₉ = (px * qy - qx * py) / Δ
        shape_function_coefficients[(i, j, k)] = (s₁, s₂, s₃, s₄, s₅, s₆, s₇, s₈, s₉)
    end
    has_ghost || delete_ghost_triangles!(tri)
    return FVMGeometry(tri, stats, cv_volumes, shape_function_coefficients)
end

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

    a(x, y, t, u)

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

    a(x, y, t, u)

where `(x, y)` is the point, `t` is the current time, and `u` is the
solution at the point `(x, y)` at time `t`, as above, with an extra 
argument `p` for additional parameters.
""" Dirichlet

get_dual_arg_types(::Type{T}, ::Type{U}, ::Type{P}) where {T,U,P} = (
    Tuple{T,T,T,U,P},                       # Typical signature 
    Tuple{T,T,T,dualgen(U),P},              # Signature with "u" a Dual 
    Tuple{T,T,dualgen(T),U,P},              # Signature with "t" a Dual 
    Tuple{T,T,dualgen(T),dualgen(U),P})     # Signature with "u" and "t" Duals
get_dual_ret_types(::Type{U}, ::Type{T}) where {U,T} = (U, dualgen(U), dualgen(T), dualgen(promote_type(U, T)))
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
    InternalConditions(functions::Tuple;
        edge_conditions::Dict{NTuple{2,Int},Int}=Dict{NTuple{2,Int},Int}(),
        point_conditions::Dict{Int,Tuple{ConditionType,Int}}=Dict{Int,Tuple{ConditionType,Int}}(),
        parameters::Tuple=ntuple(_ -> nothing, length(functions)),
        u_type=Float64,
        float_type=Float64)

This is a constructor for the [`InternalConditions`](@ref) struct, which holds the internal conditions for the PDE.
See also [`Conditions`](@ref) (which [`FVMProblem`](@ref) wraps this into), [`ConditionType`](@ref), and [`BoundaryConditions`](@ref).

# Arguments
- `functions::Tuple`

The functions that define the internal conditions. These are the functions refereed to in `edge_conditions` and `point_conditions`.

# Keyword Arguments
- `edge_conditions::Dict{NTuple{2,Int},Int}=Dict{NTuple{2,Int},Int}()`

A `Dict` that maps an oriented edge `(u, v)`, with vertices referring to points in the associated triangulation,
to the index `idx` of the associated condition function and parameters in `functions` and `parameters`. The enforced
condition on these edges are [`Neumann`](@ref) conditions.
- `point_conditions::Dict{Int,Tuple{ConditionType,Int}}=Dict{Int,Tuple{ConditionType,Int}}()`

A `Dict` that maps a vertex `u`, referring to a point in the associated triangulation, to a `Tuple` of the form
`(ConditionType, idx)`, where `ConditionType` is either [`Dudt`](@ref) or [`Dirichlet`](@ref) from [`ConditionType`](@ref),
and `idx` is the index of the associated condition function and parameters in `functions` and `parameters`.
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
    edge_conditions::Dict{NTuple{2,Int},Int}
    point_conditions::Dict{Int,Tuple{ConditionType,Int}}
    functions::F
    parameters::P
    function InternalConditions(edge_conditions, point_conditions, functions, parameters)
        @assert length(functions) == length(parameters) "The number of functions and parameters must be the same."
        return new(edge_conditions, point_conditions, functions, parameters)
    end
end

function BoundaryConditions(mesh::FVMGeometry, functions::Tuple, types::Tuple;
    parameters::Tuple=ntuple(_ -> nothing, length(functions)),
    u_type=Float64,
    float_type=Float64)
    nbnd_idx = DelaunayTriangulation.num_ghost_vertices(mesh.triangulation_statistics)
    @assert length(functions) == nbnd_idx "The number of boundary conditions must be the same as the number of parts of the mesh's boundary."
    wrapped_functions = wrap_functions(functions, parameters)
    return BoundaryConditions(wrapped_functions, parameters, types)
end

function InternalConditions(functions::Tuple=();
    edge_conditions::Dict{NTuple{2,Int},Int}=Dict{NTuple{2,Int},Int}(),
    point_conditions::Dict{Int,Tuple{ConditionType,Int}}=Dict{Int,Tuple{ConditionType,Int}}(),
    parameters::Tuple=ntuple(_ -> nothing, length(functions)),
    u_type=Float64,
    float_type=Float64)
    wrapped_functions = wrap_functions(functions, parameters)
    return InternalConditions(wrapped_functions, parameters, types)
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
- `neumann_edges::Dict{Int,Int}`

A `Dict` that maps a vertex `u` to another vertex `v`, for all ordered edges `(u, v)` from 
`edge_conditions`. This is needed so that we can easily handle incompatible boundary conditions, by 
comparing vertices from here to those in `point_conditions`. This part of the struct is not public API.
"""
struct Conditions{F<:Tuple,P<:Tuple}
    edge_conditions::Dict{NTuple{2,Int},Int}
    point_conditions::Dict{Int,Tuple{ConditionType,Int}}
    functions::F
    parameters::P
    neumann_edges::Dict{Int,Int}
end

function prepare_conditions(mesh::FVMGeometry, bc::BoundaryConditions, ic::InternalConditions)
    bc_functions = bc.functions
    bc_parameters = bc.parameters
    bc_conditions = bc.condition_types
    ic_functions = ic.functions
    ic_parameters = ic.parameters
    ic_edge_conditions = copy(ic.edge_conditions)
    ic_point_conditions = copy(ic.point_conditions)
    neumann_edges = Dict{Int,Int}()
    ne = DelaunayTriangulation.num_constrained_edges(mesh.triangulation_statistics)
    nv = DelaunayTriangulation.num_solid_vertices(mesh.triangulation_statistics)
    sizehint!(ic_edge_conditions, ne + length(ic_edge_conditions))
    sizehint!(ic_point_conditions, nv + length(ic_point_conditions))
    sizehint!(neumann_edges, ne)
    functions = (ic_functions..., bc_functions...)
    parameters = (ic_parameters..., bc_parameters...)
    conditions = Conditions(ic_edge_conditions, ic_point_conditions, functions, parameters, neumann_edges)
    return conditions
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
function get_edge_condition(conditions::Conditions, tri, i, j, bc_number, nif)
    prev_boundary_index = get_adjacent(tri, j, i)
    cur_condition = bc_conditions[nif+bc_number]
    prev_condition = bc_conditions[-prev_boundary_index+nif]
    merged_condition, replaced = reconcile_conditions(cur_condition, prev_condition)
    return merged_condition, replaced ? -prev_boundary_index + nif : nif + bc_number
end

function add_bc_number!(conditions::Conditions, condition, bc_number, i, j)
    if condition == Neumann
        neumann_edges[i] = j
        edge_conditions[(i, j)] = bc_number
    else
        point_conditions[i] = (condition, bc_number)
    end
    return nothing
end

function merge_conditions!(conditions::Conditions, mesh::FVMGeometry, nif)
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
        u = DelaunayTriangulation.get_left_boundary_node(tri, v, -bc_number)
        merged_condition, updated_bc_number = get_edge_condition(conditions, tri, u, v, bc_number, nif)
        w = get_boundary_nodes(bn_nodes, 2)
        add_condition!(conditions, merged_condition, updated_bc_number, v, w)
        for i in 2:nedges
            u = v
            v = get_boundary_nodes(bn_nodes, i)
            w = get_boundary_nodes(bn_nodes, i + 1)
            merged_condition, updated_bc_number = get_edge_condition(conditions, tri, u, v, bc_number, nif)
            add_condition!(conditions, merged_condition, updated_bc_number, v, w)
        end
    end
    hasbnd || unlock_convex_hull!(tri)
    has_ghost || delete_ghost_triangles!(tri)
    return conditions
end

function Conditions(mesh::FVMGeometry, bc::BoundaryConditions, ic::InternalConditions=InternalConditions())
    conditions = prepare_conditions(mesh, bc, ic)
    merge_conditions!(conditions, mesh, length(ic.functions))
    return conditions
end

"""
    FVMProblem(mesh, boundary_conditions[, internal_conditions];
        diffusion_function=nothing,
        diffusion_parameters=nothing,
        source_function=nothing,
        source_parameters=nothing,
        flux_function=nothing,
        flux_parameters=nothing,
        initial_condition,
        initial_time=0.0,
        final_time,
        steady=false)

Constructs an `FVMProblem`.

# Arguments 
- `mesh::FVMGeometry`

The mesh on which the PDE is defined, given as a [`FVMGeometry`](@ref).
- `boundary_conditions::BoundaryConditions`

The boundary conditions for the PDE, given as a [`BoundaryConditions`](@ref).
- `internal_conditions::InternalConditions`

The internal conditions for the PDE, given as an [`InternalConditions`](@ref). This does not 
need to be provided.

# Keyword Arguments
- `diffusion_function=nothing`

If `isnothing(flux_function)`, then this can be provided to give the diffusion-source formulation. See also [`construct_flux_function`](@ref). Should be of the form `D(x, y, t, u, p)`.
- `diffusion_parameters=nothing`

The argument `p` for `diffusion_function`.
- `source_function=nothing`

The source term, given in the form `S(x, y, t, u, p)`.
- `source_parameters=nothing`

The argument `p` for `source_function`.
- `flux_function=nothing`

The flux function, given in the form `q(x, y, t, α, β, γ, p) ↦ (qx, qy)`, where `(qx, qy)` is the flux vector, `(α, β, γ)` are the shape function coefficients for estimating `u ≈ αx+βy+γ`. If `isnothing(flux_function)`, then `diffusion_function` is used instead to construct the function.
- `flux_parameters=nothing`

The argument `p` for `flux_function`.
- `initial_condition`

The initial condition, with `initial_condition[i]` the initial value at the `i`th node of the `mesh`.
- `initial_time=0.0`

The initial time.
- `final_time`

The final time.
- `steady=false`

Whether the problem is steady or not, meaning `∂u/∂t = 0`. If the problem is steady, then the initial estimate used for the nonlinear solver that finds the steady state is given by the initial condition, and `final_time` is set to `∞`.

# Outputs
The returned value is the corresponding [`FVMProblem`](@ref) struct.
"""
struct FVMProblem{FG,BC,F,FP,R,RP,IC,FT}
    mesh::FG
    conditions::BC
    flux_function::F
    flux_parameters::FP
    source_function::R
    source_parameters::RP
    initial_condition::IC
    initial_time::FT
    final_time::FT
    steady::Bool
end

function FVMProblem(mesh::FVMGeometry, boundary_conditions::BoundaryConditions, internal_conditions::InternalConditions=InternalConditions();
    diffusion_function=nothing,
    diffusion_parameters=nothing,
    source_function=nothing,
    source_parameters=nothing,
    flux_function=nothing,
    flux_parameters=nothing,
    initial_condition,
    initial_time=0.0,
    final_time,
    steady=false)
    updated_flux_fnc = construct_flux_function(flux_function, diffusion_function, diffusion_parameters)
    conditions = Conditions(mesh, boundary_conditions, internal_conditions)
    return FVMProblem(mesh, conditions,
        updated_flux_fnc, flux_parameters,
        source_function, source_parameters,
        initial_condition, initial_time, final_time, steady)
end

"""
    construct_flux_function(q, D, Dp)

If `isnothing(q)`, then this returns the flux function based on the diffusion function `D` and 
diffusion parameters `Dp`, so that the new function is 

    (x, y, t, α, β, γ, _) -> -D(x, y, t, α*x + β*y + γ, Dp)[α, β]

Otherwise, just returns `q` again.
"""
function construct_flux_function(q, D, Dp)
    if isnothing(q)
        flux_function = let D = D, Dp = Dp
            (flx, x, y, t, α, β, γ, p) -> begin
                u = α * x + β * y + γ
                Dval = D(x, y, t, u, Dp)
                qx = -Dval * α
                qy = -Dval * β
                return (qx, qy)
            end
        end
        return flux_function
    else
        return q
    end
end

function get_shape_function_coefficients(prob, T, u)
    i, j, k = indices(T)
    s₁, s₂, s₃, s₄, s₅, s₆, s₇, s₈, s₉ = mesh.shape_function_coefficients[(i, j, k)]
    α = s₁ * u[i] + s₂ * u[j] + s₃ * u[k]
    β = s₄ * u[i] + s₅ * u[j] + s₆ * u[k]
    γ = s₇ * u[i] + s₈ * u[j] + s₉ * u[k]
    return α, β, γ
end

function get_centroid_and_edge_midpoints(prob::FVMProblem, T)
    mesh = prob.mesh
    stats = mesh.triangulation_statistics
    centroid = get_centroid(stats, T)
    midpoints = get_edge_midpoints(stats, T)
    cx, cy = getxy(centroid)
    m₁, m₂, m₃ = midpoints
    m₁x, m₁y = getxy(m₁)
    m₂x, m₂y = getxy(m₂)
    m₃x, m₃y = getxy(m₃)
    return (cx, cy), ((m₁x, m₁y), (m₂x, m₂y), (m₃x, m₃y))
end

function get_control_volume_edge_midpoints(cx, cy, m₁x, m₁y, m₂x, m₂y, m₃x, m₃y)
    m₁cx, m₁cy = (m₁x + cx) / 2, (m₁y + cy) / 2
    m₂cx, m₂cy = (m₂x + cx) / 2, (m₂y + cy) / 2
    m₃cx, m₃cy = (m₃x + cx) / 2, (m₃y + cy) / 2
    return (m₁cx, m₁cy), (m₂cx, m₂cy), (m₃cx, m₃cy)
end

function get_control_volume_edge_normals(cx, cy, m₁x, m₁y, m₂x, m₂y, m₃x, m₃y)
    e₁x, e₁y = cx - m₁x, cy - m₁y
    e₂x, e₂y = cx - m₂x, cy - m₂y
    e₃x, e₃y = cx - m₃x, cy - m₃y
    n₁x, n₁y = e₁y, -e₁x
    n₂x, n₂y = e₂y, -e₂x
    n₃x, n₃y = e₃y, -e₃x
    return (n₁x, n₁y), (n₂x, n₂y), (n₃x, n₃y)
end

function get_fluxes(prob, T, m₁cx, m₁cy, m₂cx, m₂cy, m₃cx, m₃cy, n₁x, n₁y, n₂x, n₂y, n₃x, n₃y, α, β, γ)
    i, j, k = indices(T)
    q = prob.flux_function
    p = prob.flux_parameters 
    conditions = prob.conditions
    qx₁, qy₁ = q(m₁cx, m₁cy, t, α, β, γ, p)
    qx₂, qy₂ = q(m₂cx, m₂cy, t, α, β, γ, p)
    qx₃, qy₃ = q(m₃cx, m₃cy, t, α, β, γ, p)
    qn₁ = qx₁ * n₁x + qy₁ * n₁y
    qn₂ = qx₂ * n₂x + qy₂ * n₂y
    qn₃ = qx₃ * n₃x + qy₃ * n₃y
    return qn₁, qn₂, qn₃
end

function fvm_eqs!(du::AbstractVector{T}, u, prob, t) where {T}
    fill!(du, zero(T))
    mesh = prob.mesh
    tri = mesh.triangulation
    stats = mesh.triangulation_statistics
    conditions = prob.conditions
    point_conditions = conditions.point_conditions
    point_condition_indices = keys(point_conditions)
    edge_conditions = conditions.edge_conditions

    for T in each_solid_triangle(tri)
        i, j, k = indices(T)
        α, β, γ = get_shape_function_coefficients(prob, T, u)
        (cx, cy), ((m₁x, m₁y), (m₂x, m₂y), (m₃x, m₃y)) = get_centroid_and_edge_midpoints(prob, T)
        (m₁cx, m₁cy), (m₂cx, m₂cy), (m₃cx, m₃cy) = get_control_volume_edge_midpoints(cx, cy, m₁x, m₁y, m₂x, m₂y, m₃x, m₃y)
        (n₁x, n₁y), (n₂x, n₂y), (n₃x, n₃y) = get_control_volume_edge_normals(cx, cy, m₁x, m₁y, m₂x, m₂y, m₃x, m₃y)
        summand₁, summand₂, summand₃ = get_fluxes(prob, m₁cx, m₁cy, m₂cx, m₂cy, m₃cx, m₃cy, n₁x, n₁y, n₂x, n₂y, n₃x, n₃y, α, β, γ)
        i_is_pc = i ∈ point_condition_indices
        j_is_pc = j ∈ point_condition_indices
        k_is_pc = k ∈ point_condition_indices
        i_is_neumann 
        # (i, 1), (j, 2)
        i ∉ point_condition_indices && (du[i] += summand₁)
        j ∉ point_condition_indices && (du[j] -= summand₁)

        # (j, 2), (k, 3)
        j ∉ point_condition_indices && (du[j] += summand₂)
        k ∉ point_condition_indices && (du[k] -= summand₂)

        # (k, 3), (i, 1)
        k ∉ point_condition_indices && (du[k] += summand₃)
        i ∉ point_condition_indices && (du[i] -= summand₃)
    end
end

struct Conditions{F<:Tuple,P<:Tuple}
    edge_conditions::Dict{NTuple{2,Int},Int}
    point_conditions::Dict{Int,Tuple{ConditionType,Int}}
    functions::F
    parameters::P
    neumann_edges::Dict{Int,Int}
end

curve_1 = [
    [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0), (15.0, 0.0), (20.0, 0.0), (25.0, 0.0)],
    [(25.0, 0.0), (25.0, 5.0), (25.0, 10.0), (25.0, 15.0), (25.0, 20.0), (25.0, 25.0)],
    [(25.0, 25.0), (20.0, 25.0), (15.0, 25.0), (10.0, 25.0), (5.0, 25.0), (0.0, 25.0)],
    [(0.0, 25.0), (0.0, 20.0), (0.0, 15.0), (0.0, 10.0), (0.0, 5.0), (0.0, 0.0)]
] # outer-most boundary: counter-clockwise  
curve_2 = [
    [(4.0, 6.0), (4.0, 14.0), (4.0, 20.0), (18.0, 20.0), (20.0, 20.0)],
    [(20.0, 20.0), (20.0, 16.0), (20.0, 12.0), (20.0, 8.0), (20.0, 4.0)],
    [(20.0, 4.0), (16.0, 4.0), (12.0, 4.0), (8.0, 4.0), (4.0, 4.0), (4.0, 6.0)]
] # inner boundary: clockwise 
curve_3 = [
    [(12.906, 10.912), (16.0, 12.0), (16.16, 14.46), (16.29, 17.06),
    (13.13, 16.86), (8.92, 16.4), (8.8, 10.9), (12.906, 10.912)]
] # this is inside curve_2, so it's counter-clockwise 
curves = [curve_1, curve_2, curve_3]
points = [
    (3.0, 23.0), (9.0, 24.0), (9.2, 22.0), (14.8, 22.8), (16.0, 22.0),
    (23.0, 23.0), (22.6, 19.0), (23.8, 17.8), (22.0, 14.0), (22.0, 11.0),
    (24.0, 6.0), (23.0, 2.0), (19.0, 1.0), (16.0, 3.0), (10.0, 1.0), (11.0, 3.0),
    (6.0, 2.0), (6.2, 3.0), (2.0, 3.0), (2.6, 6.2), (2.0, 8.0), (2.0, 11.0),
    (5.0, 12.0), (2.0, 17.0), (3.0, 19.0), (6.0, 18.0), (6.5, 14.5),
    (13.0, 19.0), (13.0, 12.0), (16.0, 8.0), (9.8, 8.0), (7.5, 6.0),
    (12.0, 13.0), (19.0, 15.0)
]
boundary_nodes, points = convert_boundary_points_to_indices(curves; existing_points=points)
cons_tri = triangulate(points; boundary_nodes=boundary_nodes, check_arguments=false)
