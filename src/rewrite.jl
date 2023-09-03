export FVMGeometry,
    FVMProblem,
    FVMSystem,
    SteadyFVMProblem,
    BoundaryConditions,
    InternalConditions,
    Neumann,
    Dudt,
    Dirichlet,
    solve

# Properties of a control volume's intersection with a triangle
struct TriangleProperties
    shape_function_coefficients::NTuple{9,Float64}
    cv_edge_midpoints::NTuple{3,NTuple{2,Float64}}
    cv_normals::NTuple{3,NTuple{2,Float64}}
    cv_edge_lengths::NTuple{3,Float64}
end

"""
    FVMGeometry(tri::Triangulation)

This is a constructor for the [`FVMGeometry`](@ref) struct, which holds the mesh and associated data for the PDE.

It is assumed that all vertices in `tri` are in the triangulation, meaning `v` is in `tri` for each `v` in `each_point_index(tri)`.
"""
struct FVMGeometry{T,S}
    triangulation::T
    triangulation_statistics::S
    cv_volumes::Vector{Float64}
    triangle_props::Dict{NTuple{3,Int},TriangleProperties}
end
function FVMGeometry(tri::Triangulation)
    has_ghost = DelaunayTriangulation.has_ghost_triangles(tri)
    has_ghost || add_ghost_triangles!(tri)
    stats = statistics(tri)
    nn = DelaunayTriangulation.num_solid_vertices(stats)
    nt = DelaunayTriangulation.num_solid_triangles(stats)
    cv_volumes = zeros(Int, nn)
    triangle_props = Dict{NTuple{3,Int},TriangleProperties}()
    sizehint!(cv_volumes, nn)
    sizehint!(triangle_props, nt)
    for T in each_solid_triangle(tri)
        i, j, k = indices(T)
        p, q, r = get_point(tri, i, j, k)
        px, py = getxy(p)
        qx, qy = getxy(q)
        rx, ry = getxy(r)
        ## Get the centroid of the triangle, and the midpoint of each edge
        centroid = DelaunayTriangulation.get_centroid(stats, T)
        m1, m2, m3 = DelaunayTriangulation.get_edge_midpoints(stats, T)
        ## Need to get the sub-control volume areas
        # We need to connect the centroid to each vertex 
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
        ## Next, we need to compute the shape function coefficients
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
        shape_function_coefficients = (s₁, s₂, s₃, s₄, s₅, s₆, s₇, s₈, s₉)
        ## Now we need the control volume edge midpoints 
        m₁cx, m₁cy = (m₁x + cx) / 2, (m₁y + cy) / 2
        m₂cx, m₂cy = (m₂x + cx) / 2, (m₂y + cy) / 2
        m₃cx, m₃cy = (m₃x + cx) / 2, (m₃y + cy) / 2
        ## Next, we need the normal vectors to the control volume edges 
        e₁x, e₁y = cx - m₁x, cy - m₁y
        e₂x, e₂y = cx - m₂x, cy - m₂y
        e₃x, e₃y = cx - m₃x, cy - m₃y
        ℓ₁ = norm((e₁x, e₁y))
        ℓ₂ = norm((e₂x, e₂y))
        ℓ₃ = norm((e₃x, e₃y))
        n₁x, n₁y = e₁y, -e₁x
        n₂x, n₂y = e₂y, -e₂x
        n₃x, n₃y = e₃y, -e₃x
        ## Now construct the TriangleProperties
        triangle_props[indices(T)] = TriangleProperties(shape_function_coefficients, ((m₁cx, m₁cy), (m₂cx, m₂cy), (m₃cx, m₃cy)), ((n₁x, n₁y), (n₂x, n₂y), (n₃x, n₃y)), (ℓ₁, ℓ₂, ℓ₃))
    end
    has_ghost || delete_ghost_triangles!(tri)
    return FVMGeometry(tri, stats, cv_volumes, triangle_props)
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

# Outputs
The returned value is the corresponding [`FVMProblem`](@ref) struct. You can then solve the problem using `solve` from DifferentialEquations.jl.
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
    SteadyFVMProblem{P<:Union{<:FVMProblem,<:FVMSystem}}

This is a wrapper for [`FVMProblem`](@ref) or [`FVMSystem`](@ref) that indicates that the problem is to be solved as a 
steady-state problem. You can then solve the problem using `solve` from (Simple)NonlinearSolve.jl.
"""
struct SteadyFVMProblem{P}
    problem::P
end

"""
    FVMSystem{N,FG,P,IC,FT,FT}

Representation of a system of PDEs. The constructor for this struct is 

    FVMSystem(prob1, prob2, ..., probN),

where each `probi` is a [`FVMProblem`](@ref) for the `i`th component of the system.
For these [`FVMProblem`](@ref)s, the functions involved, such as the condition functions, should 
all be defined so that the `u` argument assumes the form `u = (u₁, u₂, ..., uN)` (both `Tuple`s and `Vector`s will be passed), 
where `uᵢ` is the solution for the `i`th component of the system. For the flux functions, 
which for a [`FVMProblem`](@ref) takes the form

    q(x, y, t, α, β, γ, p) ↦ (qx, qy),

the same form is used, except `α`, `β`, `γ` are all `Tuple`s so that `α[i]*x + β[i]*y + γ` is the 
approximation to `uᵢ`.

This problem is solved in the same way as a [`FVMProblem`](@ref), except the problem is defined such that 
the solution returns a matrix at each time, where the `(j, i)`th component corresponds to the solution at the `i`th 
node for the `j`th component.
"""
struct FVMSystem{N,FG,P,IC,FT,FT}
    mesh::FG
    problems::P
    initial_condition::IC
    initial_time::FT
    final_time::FT
    function FVMSystem(mesh::FG, problems::P, initial_condition::IC, initial_time::FT, final_time::FT) where {FG,P,IC,FT}
        @assert length(problems) > 0 "There must be at least one problem."
        @assert all(p -> p.mesh === mesh, problems) "All problems must have the same mesh."
        @assert all(p -> p.initial_time === initial_time, problems) "All problems must have the same initial time."
        @assert all(p -> p.final_time === final_time, problems) "All problems must have the same final time."
        @assert size(initial_condition) == (length(problems), length(initial_condition[1])) "The initial condition must be a matrix with the same number of rows as the number of problems."
        return FVMSystem{length(problems),FG,P,IC,FT,FT}(mesh, problems, initial_condition, initial_time, final_time)
    end
end
function FVMSystem(probs::Vararg{FVMProblem,N}) where {N}
    N == 0 && error("There must be at least one problem.")
    mesh = probs[1].mesh
    initial_time = probs[1].initial_time
    final_time = probs[1].final_time
    n = DelaunayTriangulation.num_solid_vertices(mesh)
    ic₁ = probs[1].initial_condition
    initial_condition = similar(ic₁, N, length(ic₁))
    for (i, prob) in enumerate(probs)
        initial_condition[i, :] .= prob.initial_condition
    end
    return FVMSystem(mesh, probs, initial_condition, initial_time, final_time)
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
            (x, y, t, α, β, γ, p) -> begin
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

function idx_is_point_condition(prob::FVMProblem, idx)
    return idx ∈ keys(prob.conditions.point_conditions)
end
function idx_is_point_condition(prob::FVMSystem{N}, idx, var) where {N}
    return idx_is_point_condition(prob.problens[var], idx)
end
function idx_is_dudt_condition(prob::FVMProblem, idx)
    return prob.conditions.point_conditions[idx][1] == Dudt
end
function idx_is_dudt_condition(prob::FVMSystem{N}, idx, var) where {N}
    return idx_is_dudt_condition(prob.problems[var], idx)
end
function idx_is_neumann_condition(prob::FVMProblem, i, j)
    return (i, j) ∈ keys(prob.conditions.edge_conditions)
end
function idx_is_neumann_condition(prob::FVMSystem{N}, i, j, var) where {N}
    return idx_is_neumann_condition(prob.problems[var], i, j)
end

function get_shape_function_coefficients(props::TriangleProperties, T, u, ::FVMProblem)
    i, j, k = indices(T)
    s₁, s₂, s₃, s₄, s₅, s₆, s₇, s₈, s₉ = props.shape_function_coefficients
    α = s₁ * u[i] + s₂ * u[j] + s₃ * u[k]
    β = s₄ * u[i] + s₅ * u[j] + s₆ * u[k]
    γ = s₇ * u[i] + s₈ * u[j] + s₉ * u[k]
    return α, β, γ
end
function get_shape_function_coefficients(props::TriangleProperties, T, u, ::FVMSystem{N})
    i, j, k = indices(T)
    s₁, s₂, s₃, s₄, s₅, s₆, s₇, s₈, s₉ = props.shape_function_coefficients
    α = ntuple(ℓ -> s₁ * u[ℓ, i] + s₂ * u[ℓ, j] + s₃ * u[ℓ, k], N)
    β = ntuple(ℓ -> s₄ * u[ℓ, i] + s₅ * u[ℓ, j] + s₆ * u[ℓ, k], N)
    γ = ntuple(ℓ -> s₇ * u[ℓ, i] + s₈ * u[ℓ, j] + s₉ * u[ℓ, k], N)
    return α, β, γ
end

function evaluate_flux()

end

function get_flux(prob::FVMProblem, props, α, β, γ, t, i, j, edge_index)
    # For checking if an edge is Neumann, we need only check e.g. (i, j) and not (j, i), since we do not allow for internal Neumann edges.
    ij_is_neumann = idx_is_neumann_condition(prob, i, j)
    x, y = props.control_volume_midpoints[edge_index]
    nx, ny = props.control_volume_normals[edge_index]
    ℓ = props.control_volume_edge_lengths[edge_index]
    if !ij_is_neumann
        qx, qy = prob.flux_function(x, y, t, α, β, γ, prob.flux_parameters)
        qn = qx * nx + qy * ny
    else
        function_index = prob.conditions.edge_conditions[(i, j)]
        a, ap = prob.conditions.functions[function_index], prob.conditions.parameters[function_index]
        qn = a(x, y, t, α * x + β * y + γ, ap)
    end
    return qn * ℓ
end
function get_flux(prob::FVMSystem{N}, props, α, β, γ, t, i, j, edge_index)
    x, y = props.control_volume_midpoints[edge_index]
    nx, ny = props.control_volume_normals[edge_index]
    ℓ = props.control_volume_edge_lengths[edge_index]
    u_shape = ntuple(ℓ -> α[ℓ] * x + β[ℓ] * y + γ[ℓ], N)
    qn = ntuple(N) do ℓ
        ij_is_neumann = idx_is_neumann_condition(prob, i, j, ℓ)
        if !ij_is_neumann
            qx, qy = prob.problems[ℓ].flux_function(x, y, t, α, β, γ, prob.problems[ℓ].flux_parameters)
            qn = qx * nx + qy * ny
        else
            function_index = prob.problems[ℓ].conditions.edge_conditions[(i, j)]
            a, ap = prob.problems[ℓ].conditions.functions[function_index], prob.problems[ℓ].conditions.parameters[function_index]
            qn = a(x, y, t, u_shape, ap)
        end
        return qn * ℓ
    end
    return qn
end

function get_fluxes(prob, props, α, β, γ, t)
    q1 = get_flux(prob, props, α, β, γ, t, i, j, 1)
    q2 = get_flux(prob, props, α, β, γ, t, j, k, 2)
    q3 = get_flux(prob, props, α, β, γ, t, k, i, 3)
    return q1, q2, q3
end

function update_du!(du, prob::FVMProblem, i, j, k, summand₁, summand₂, summand₃)
    i_is_pc = idx_is_point_condition(prob, i)
    j_is_pc = idx_is_point_condition(prob, j)
    k_is_pc = idx_is_point_condition(prob, k)
    if !i_is_pc
        du[i] -= summand₁
        du[j] += summand₁
    end
    if !j_is_pc
        du[j] -= summand₂
        du[k] += summand₂
    end
    if !k_is_pc
        du[k] -= summand₃
        du[i] += summand₃
    end
end
function update_du!(du, prob::FVMSystem{N}, i, j, k, ℓ, summand₁, summand₂, summand₃) where {N}
    i_is_pc = idx_is_point_condition(prob, i, ℓ)
    j_is_pc = idx_is_point_condition(prob, j, ℓ)
    k_is_pc = idx_is_point_condition(prob, k, ℓ)
    @. begin
        if !i_is_pc
            du[:, i] -= summand₁
            du[:, j] += summand₁
        end
        if !j_is_pc
            du[:, j] -= summand₂
            du[:, k] += summand₂
        end
        if !k_is_pc
            du[:, k] -= summand₃
            du[:, i] += summand₃
        end
    end
end

function fvm_eqs_single_triangle!(du, u, prob, t, T)
    i, j, k = indices(T)
    props = prob.mesh.triangle_props[(i, j, k)]
    α, β, γ = get_shape_function_coefficients(props, T, u, prob)
    summand₁, summand₂, summand₃ = get_fluxes(prob, props, α, β, γ, t)
    update_du!(du, prob, i, j, k, summand₁, summand₂, summand₃)
    return nothing
end

function fvm_eqs_single_source_contribution!(du, u, prob::FVMProblem, t, i)
    p = get_point(prob.mesh.triangulation, i)
    x, y = getxy(p)
    if !idx_is_point_condition(prob, i)
        du[i] += prob.source_function(x, y, t, u[i], prob.source_parameters)
    elseif idx_is_dudt_condition(prob, i)
        function_index = prob.conditions.point_conditions[i][2]
        a, ap = prob.conditions.functions[function_index], prob.conditions.parameters[function_index]
        du[i] = a(x, y, t, u[i], ap)
    else # Dirichlet
        du[i] = zero(eltype(du))
    end
    return nothing
end
function fvm_eqs_single_source_contribution!(du, u, prob::FVMSystem{N}, t, i) where {N}
    p = get_point(prob.mesh.triangulation, i)
    x, y = getxy(p)
    uᵢ = @views u[:, i]
    for j in 1:N
        if !idx_is_point_condition(prob, i, j)
            du[j, i] += prob.problems[j].source_function(x, y, t, uᵢ, prob.problems[j].source_parameters)
        elseif idx_is_dudt_condition(prob, i, j)
            function_index = prob.problems[j].conditions.point_conditions[i][2]
            a, ap = prob.problems[j].conditions.functions[function_index], prob.problems[j].conditions.parameters[function_index]
            du[j, i] = a(x, y, t, uᵢ, ap)
        else # Dirichlet
            du[j, i] = zero(eltype(du))
        end
    end
end

function serial_fvm_eqs!(du, u, prob, t)
    fill!(du, zero(eltype(du)))
    for T in each_solid_triangle(prob.mesh.triangulation)
        fvm_eqs_single_triangle!(du, u, prob, t, T)
    end
    for i in each_solid_vertex(prob.mesh.triangulation)
        fvm_eqs_single_source_contribution!(du, u, prob, t, i)
    end
    return nothing
end

function combine_duplicated_du!(du, duplicated_du, prob)
    if prob isa FVMProblem
        for _du in eachcol(duplicated_du)
            du .+= _du
        end
    else
        for i in axes(duplicated_du, 3)
            du .+= duplicated_du[:, :, i]
        end
    end
    return nothing
end

function parallel_fvm_eqs!(du, u, p, t)
    duplicated_du, solid_triangles,
    solid_vertices, chunked_solid_triangles,
    prob = p.duplicated_du, p.solid_triangles,
    p.solid_vertices, p.chunked_solid_triangles,
    p.prob
    fill!(du, zero(eltype(du)))
    _duplicated_du = get_tmp(duplicated_du, du)
    fill!(_duplicated_du, zero(eltype(du)))
    Threads.@threads for (triangle_range, chunk_idx) in chunked_solid_triangles
        for triangle_idx in triangle_range
            T = solid_triangles[triangle_idx]
            if prob isa FVMProblem
                @views fvm_eqs_single_triangle!(_duplicated_du[:, chunk_idx], u, prob, t, T)
            else
                @views fvm_eqs_single_triangle!(_duplicated_du[:, :, chunk_idx], u, prob, t, T)
            end
        end
    end
    combine_duplicated_du!(du, duplicated_du, prob)
    Threads.@threads for i in solid_vertices
        fvm_eqs_single_source_contribution!(du, u, prob, t, i)
    end
    return nothing
end

function fvm_eqs!(du, u, p, t)
    prob, parallel = p.prob, p.parallel
    if parallel == Val(false)
        return serial_fvm_eqs!(du, u, prob, t)
    else
        return parallel_fvm_eqs!(du, u, p, t)
    end
end

function update_dirichlet_nodes_single!(u, t, prob::FVMProblem, i, condition, function_index)
    if condition == Dirichlet
        a, ap = prob.conditions.functions[function_index], prob.conditions.parameters[function_index]
        p = get_point(prob.mesh.triangulation, i)
        x, y = getxy(p)
        u[i] = a(x, y, t, u[i], ap)
    end
    return nothing
end
function update_dirichlet_nodes_single!(u, t, prob::FVMSystem{N}, i, j, condition, function_index) where {N}
    if condition == Dirichlet
        a, ap = prob.problems[j].conditions.functions[function_index], prob.problems[j].conditions.parameters[function_index]
        p = get_point(prob.mesh.triangulation, i)
        x, y = getxy(p)
        @views u[j, i] = a(x, y, t, u[:, i], ap)
    end
end

function serial_update_dirichlet_nodes!(u, t, prob::FVMProblem)
    for (i, (condition, function_index)) in prob.conditions.point_conditions
        update_dirichlet_nodes_single!(u, t, prob, i, condition, function_index)
    end
    return nothing
end
function serial_update_dirichlet_nodes!(u, t, prob::FVMSystem{N}) where {N}
    for j in 1:N
        for (i, (condition, function_index)) in prob.problems[j].conditions.point_conditions
            update_dirichlet_nodes_single!(u, t, prob, i, j, condition, function_index)
        end
    end
end

function parallel_update_dirichlet_nodes!(u, t, p, prob::FVMProblem)
    point_conditions = p.point_conditions
    Threads.@threads for i in point_conditions
        point_conditions_dict = prob.conditions.point_conditions
        condition, function_index = point_conditions_dict[i]
        update_dirichlet_nodes_single!(u, t, prob, i, condition, function_index)
    end
    return nothing
end
function parallel_update_dirichlet_nodes!(u, t, p, prob::FVMSystem{N}) where {N}
    point_conditions = p.point_conditions
    for j in 1:N
        Threads.@threads for i in point_conditions[j]
            point_conditions_dict = prob.problems[j].conditions.point_conditions
            condition, function_index = point_conditions_dict[i]
            update_dirichlet_nodes_single!(u, t, prob, i, condition, function_index)
        end
    end
    return nothing
end

function update_dirichlet_nodes!(integrator)
    prob, parallel = integrator.p.prob, integrator.p.parallel
    if parallel == Val(false)
        return serial_update_dirichlet_nodes!(integrator.u, integrator.t, prob)
    else
        return parallel_update_dirichlet_nodes!(integrator.u, integrator.t, integrator.p, prob)
    end
    return nothing
end

function get_multithreading_vectors(prob::Union{FVMProblem,FVMSystem{N}}) where {N}
    u = prob.initial_condition
    nt = Threads.nthreads()
    if prob isa FVMProblem
        duplicated_du = DiffCache(similar(u, length(u), nt))
        point_conditions = collect(keys(prob.conditions.point_conditions))
    else
        duplicated_du = DiffCache(similar(u, size(u, 1), size(u, 2), nt))
        point_conditions = ntuple(i -> collect(keys(prob.problems[i].conditions.point_conditions)), N)
    end
    solid_triangles = collect(each_solid_triangle(prob.mesh.triangulation))
    solid_vertices = collect(each_solid_vertex(prob.mesh.triangulation))
    chunked_solid_triangles = chunks(solid_triangles, nt)
    return (
        duplicated_du=duplicated_du,
        point_conditions=point_conditions,
        solid_triangles=solid_triangles,
        solid_vertices=solid_vertices,
        chunked_solid_triangles=chunked_solid_triangles,
        parallel=Val(true),
        prob=prob
    )
end

"""
    jacobian_sparsity(prob::Union{FVMProblem,FVMSystem})

Constructs the sparse matrix which has the same sparsity pattern as the Jacobian for the finite volume equations 
corresponding to the [`FVMProblem`](@ref) or [`FVMSystem`](@ref) given by `prob`.
"""
function jacobian_sparsity(prob::FVMProblem)
    tri = prob.mesh.triangulation
    I = Int64[]   # row indices 
    J = Int64[]   # col indices 
    V = Float64[] # values (all 1)
    n = length(prob.initial_condition)
    sizehint!(I, 6n) # points have, on average, six neighbours in a DelaunayTriangulation
    sizehint!(J, 6n)
    sizehint!(V, 6n)
    for i in each_solid_vertex(tri)
        push!(I, i)
        push!(J, i)
        push!(V, 1.0)
        ngh = get_neighbours(tri, i)
        for j in ngh
            if !DelaunayTriangulation.is_boundary_index(j)
                push!(I, i)
                push!(J, j)
                push!(V, 1.0)
            end
        end
    end
    return sparse(I, J, V)
end
function jacobian_sparsity(prob::FVMSystem{N}) where {N}
    tri = prob.mesh.triangulation
    I = Int64[]   # row indices
    J = Int64[]   # col indices
    V = Float64[] # values (all 1)
    n = length(prob.initial_condition)
    sizehint!(I, 6n) # points have, on average, six neighbours in a DelaunayTriangulation. We don't need to multiply by N here, since length(prob.initial_condition) is actually N * num_solid_vertices(tri) already 
    sizehint!(J, 6n)
    sizehint!(V, 6n)
    for i in each_solid_vertex(tri)
        for j in 1:N
            push!(I, i)
            push!(J, (j - 1) * N + i)
            push!(V, 1.0)
        end
        ngh = get_neighbours(tri, i)
        for j in ngh
            if !DelaunayTriangulation.is_boundary_index(j)
                for k in 1:N
                    push!(I, i)
                    push!(J, (k - 1) * N + j)
                    push!(V, 1.0)
                end
            end
        end
    end
    return sparse(I, J, V)
end

@inline function dirichlet_callback(has_saveat=true)
    cb = DiscreteCallback(
        Returns(true),
        (integrator, t, u) -> update_dirichlet_nodes!(integrator); save_positions=(!has_saveat, !has_saveat)
    )
    return cb
end

function SciMLBase.ODEProblem(prob::Union{FVMProblem,FVMSystem};
    specialization::Type{S}=SciMLBase.AutoSpecialize,
    jac_prototype=jacobian_sparsity(prob),
    parallel::Bool,
    kwargs...) where {S}
    par = Val(parallel)
    initial_time = prob.initial_time
    final_time = prob.final_time
    time_span = (initial_time, final_time)
    initial_condition = prob.initial_condition
    kwarg_dict = Dict(kwargs)
    dirichlet_cb = dirichlet_callback(:saveat ∈ keys(kwarg_dict))
    if :callback ∈ keys(kwarg_dict)
        callback = CallbackSet(kwarg_dict[:callback], dirichlet_cb)
    else
        callback = CallbackSet(dirichlet_cb)
    end
    delete!(kwargs, :callback)
    f = ODEFunction{true,S}(fvm_eqs!; jac_prototype)
    p = par ? get_multithreading_vectors(prob) : prob
    ode_problem = ODEProblem{true,S}(f, initial_condition, time_span, p; callback=callback, kwargs...)
    return ode_problem
end
function SciMLBase.NonlinearProblem(prob::SteadyFVMProblem; kwargs...)
    ode_prob = ODEProblem(prob.problem; kwargs...)
    nl_prob = NonlinearProblem{true}(ode_prob.f, ode_prob.u0, ode_prob.p; kwargs...)
    return nl_prob
end

CommonSolve.init(prob::Union{FVMProblem,FVMSystem}, alg; kwargs...) = CommonSolve.init(ODEProblem(prob, kwargs...), alg; kwargs...)
CommonSolve.solve(prob::SteadyFVMProblem, alg; kwargs...) = CommonSolve.solve(NonlinearProblem(prob; kwargs...), alg; kwargs...)

@doc """
    solve(prob::Union{FVMProblem,FVMSystem}, alg; kwargs...)

Solves the given [`FVMProblem`](@ref) or [`FVMSystem`](@ref) `prob` with the algorithm `alg`, with keyword 
arguments `kwargs` passed to the solver as in DifferentialEquations.jl. The returned type for a [`FVMProblem`](@ref)
is a `sol::ODESolution`, with the `i`th component of the solution referring to the `i`th 
node in the underlying mesh, and accessed like the solutions in DifferentialEquations.jl. If `prob` is a 
[`FVMSystem`](@ref), the `(j, i)`th component of the solution instead refers to the `i`th node 
for the `j`th component of the system.
""" solve(::Union{FVMProblem,FVMSystem}, ::Any; kwargs...)

@doc """
    solve(prob::SteadyFVMProblem, alg; kwargs...)

Solves the given [`SteadyFVMProblem`](@ref) `prob` with the algorithm `alg`, with keyword
arguments `kwargs` passed to the solver as in (Simple)NonlinearSolve.jl. The returned type
is a `NonlinearSolution`, and the `i`th component of the solution if the steady state for the 
`i`th node in the underlying mesh. If the underlying problem is instead a [`FVMSystem`](@ref), 
rather than a [`FVMProblem`](@ref), it is the `(j, i)`th component that refers to the `i`th 
node of the mesh for the `j`th component of the system.
""" solve(::SteadyFVMProblem, ::Any; kwargs...)