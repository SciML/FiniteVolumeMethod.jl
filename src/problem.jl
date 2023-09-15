abstract type AbstractFVMProblem end
@inline get_dudt_fidx(prob::AbstractFVMProblem, i) = get_dudt_fidx(prob.conditions, i)
@inline get_neumann_fidx(prob::AbstractFVMProblem, i, j) = get_neumann_fidx(prob.conditions, i, j)
@inline get_dirichlet_fidx(prob::AbstractFVMProblem, i) = get_dirichlet_fidx(prob.conditions, i)
@inline get_constrained_fidx(prob::AbstractFVMProblem, i, j) = get_constrained_fidx(prob.conditions, i, j)
@inline eval_condition_fnc(prob::AbstractFVMProblem, fidx, x, y, t, u) = eval_condition_fnc(prob.conditions, fidx, x, y, t, u)
@inline eval_source_fnc(prob::AbstractFVMProblem, x, y, t, u) = prob.source_function(x, y, t, u, prob.source_parameters)
@inline is_dudt_node(prob::AbstractFVMProblem, node) = is_dudt_node(prob.conditions, node)
@inline is_neumann_edge(prob::AbstractFVMProblem, i, j) = is_neumann_edge(prob.conditions, i, j)
@inline is_dirichlet_node(prob::AbstractFVMProblem, node) = is_dirichlet_node(prob.conditions, node)
@inline is_constrained_edge(prob::AbstractFVMProblem, i, j) = is_constrained_edge(prob.conditions, i, j)
@inline has_condition(prob::AbstractFVMProblem, node) = has_condition(prob.conditions, node)
@inline has_dirichlet_nodes(prob::AbstractFVMProblem) = has_dirichlet_nodes(prob.conditions)
@inline get_triangle_props(prob::AbstractFVMProblem, i, j, k) = get_triangle_props(prob.mesh, i, j, k)
@inline DelaunayTriangulation.get_point(prob::AbstractFVMProblem, i) = get_point(prob.mesh, i)
@inline get_volume(prob::AbstractFVMProblem, i) = get_volume(prob.mesh, i)
@inline get_dirichlet_nodes(prob::AbstractFVMProblem) = get_dirichlet_nodes(prob.conditions)

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
        final_time)

Constructs an `FVMProblem`. See also [`FVMSystem`](@ref) and [`SteadyFVMProblem`](@ref).

# Arguments 
- `mesh::FVMGeometry`

The mesh on which the PDE is defined, given as a [`FVMGeometry`](@ref).
- `boundary_conditions::BoundaryConditions`

The boundary conditions for the PDE, given as a [`BoundaryConditions`](@ref).
- `internal_conditions::InternalConditions=InternalConditions()`

The internal conditions for the PDE, given as an [`InternalConditions`](@ref). This argument 
is optional.

# Keyword Arguments
- `diffusion_function=nothing`

If `isnothing(flux_function)`, then this can be provided to give the diffusion-source formulation. See also [`construct_flux_function`](@ref). Should be of the form `D(x, y, t, u, p)`.
- `diffusion_parameters=nothing`

The argument `p` for `diffusion_function`.
- `source_function=(x, y, t, u, p) -> zero(u)`

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
The returned value is the corresponding [`FVMProblem`](@ref) struct. You can then solve the problem using [`solve(::Union{FVMProblem,FVMSystem}, ::Any; kwargs...)`](@ref) from DifferentialEquations.jl.
"""
struct FVMProblem{FG,BC,F,FP,R,RP,IC,FT} <: AbstractFVMProblem
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
function Base.show(io::IO, ::MIME"text/plain", prob::FVMProblem)
    nv = DelaunayTriangulation.num_solid_vertices(prob.mesh.triangulation)
    t0 = prob.initial_time
    tf = prob.final_time
    print(io, "FVMProblem with $(nv) nodes and time span ($t0, $tf)")
end
@inline eval_flux_function(prob::FVMProblem, x, y, t, α, β, γ) = prob.flux_function(x, y, t, α, β, γ, prob.flux_parameters)

function FVMProblem(mesh::FVMGeometry, boundary_conditions::BoundaryConditions, internal_conditions::InternalConditions=InternalConditions();
    diffusion_function=nothing,
    diffusion_parameters=nothing,
    source_function=(x, y, t, u, p) -> zero(eltype(u)),
    source_parameters=nothing,
    flux_function=nothing,
    flux_parameters=nothing,
    initial_condition,
    initial_time=0.0,
    final_time)
    conditions = Conditions(mesh, boundary_conditions, internal_conditions)
    return FVMProblem(mesh, conditions;
        diffusion_function, diffusion_parameters,
        source_function, source_parameters,
        flux_function, flux_parameters,
        initial_condition, initial_time, final_time)
end
function FVMProblem(mesh::FVMGeometry, conditions::Conditions;
    diffusion_function=nothing,
    diffusion_parameters=nothing,
    source_function=(x, y, t, u, p) -> zero(eltype(u)),
    source_parameters=nothing,
    flux_function=nothing,
    flux_parameters=nothing,
    initial_condition,
    initial_time=0.0,
    final_time)
    updated_flux_fnc = construct_flux_function(flux_function, diffusion_function, diffusion_parameters)
    return FVMProblem(mesh, conditions,
        updated_flux_fnc, flux_parameters,
        source_function, source_parameters,
        initial_condition, initial_time, final_time)
end

"""
    SteadyFVMProblem(prob::AbstractFVMProblem)

This is a wrapper for an `AbstractFVMProblem` that indicates that the problem is to be solved as a steady-state problem. 
You can then solve the problem using [`solve(::SteadyFVMProblem, ::Any; kwargs...)`](@ref) from NonlinearSolve.jl. Note that you 
need to have set the final time to `Inf` if you want a steady state out at infinity rather than some finite actual time.

See also [`FVMProblem`](@ref) and [`FVMSystem`](@ref).
"""
struct SteadyFVMProblem{P<:AbstractFVMProblem,M<:FVMGeometry} <: AbstractFVMProblem
    problem::P
    mesh::M
end
function SteadyFVMProblem(prob::P) where {P}
    return SteadyFVMProblem{P,typeof(prob.mesh)}(prob, prob.mesh)
end

function Base.show(io::IO, ::MIME"text/plain", prob::SteadyFVMProblem)
    nv = DelaunayTriangulation.num_solid_vertices(prob.mesh.triangulation)
    is_sys = is_system(prob)
    if !is_sys
        print(io, "SteadyFVMProblem with $(nv) nodes")
    else
        print(io, "SteadyFVMProblem with $(nv) nodes and $(_neqs(prob)) equations")
    end
end
@inline eval_flux_function(prob::SteadyFVMProblem, x, y, t, α, β, γ) = eval_flux_function(prob.problem, x, y, t, α, β, γ)

"""
    FVMSystem(prob1, prob2, ..., probN)

Constructs a representation for a system of PDEs, where each `probi` is 
a [`FVMProblem`](@ref) for the `i`th component of the system.

For these [`FVMProblem`](@ref)s, the functions involved, such as the condition functions, should 
all be defined so that the `u` argument assumes the form `u = (u₁, u₂, ..., uN)` (both `Tuple`s and `Vector`s will be passed), 
where `uᵢ` is the solution for the `i`th component of the system. For the flux functions, 
which for a [`FVMProblem`](@ref) takes the form

    q(x, y, t, α, β, γ, p) ↦ (qx, qy),

the same form is used, except `α`, `β`, `γ` are all `Tuple`s so that `α[i]*x + β[i]*y + γ[i]` is the 
approximation to `uᵢ`. 

!!! warning "Providing default flux functions"

    Due to this difference in flux functions, and the need to provide 
    `α`, `β`, and `γ` to the flux function, for `FVMSystem`s you need to 
    provide a flux function rather than a diffusion function. If you do
    provide a diffusion function, it will error when you try to solve 
    the problem.

This problem is solved in the same way as a [`FVMProblem`](@ref), except the problem is defined such that 
the solution returns a matrix at each time, where the `(j, i)`th component corresponds to the solution at the `i`th 
node for the `j`th component.

See also [`FVMProblem`](@ref) and [`SteadyFVMProblem`](@ref).

!!! note 
    To construct a steady-state problem for an `FVMSystem`, you need to apply 
    [`SteadyFVMProblem`](@ref) to the system rather than first applying it to
    each individual [`FVMProblem`](@ref) in the system.
"""
struct FVMSystem{N,FG,P,IC,FT,F,S,FF} <: AbstractFVMProblem
    mesh::FG
    problems::P
    initial_condition::IC
    initial_time::FT
    final_time::FT
    conditions::NTuple{N,SimpleConditions}
    cnum_fncs::NTuple{N,Int} # cumulative numbers. e.g. if num_fncs is the number of functions for each variable, then cnum_fncs[i] = sum(num_fncs[1:i-1]).
    functions::F
    source_functions::S 
    flux_functions::FF
    function FVMSystem(mesh::FG, problems::P, initial_condition::IC, initial_time::FT, final_time::FT, conditions, num_fncs, functions::F, source_functions::S, flux_functions::FF) where {FG<:FVMGeometry,P,IC,FT,F,S,FF}
        @assert length(problems) > 0 "There must be at least one problem."
        @assert all(p -> p.mesh === mesh, problems) "All problems must have the same mesh."
        @assert all(p -> p.initial_time === initial_time, problems) "All problems must have the same initial time."
        @assert all(p -> p.final_time === final_time, problems) "All problems must have the same final time."
        @assert size(initial_condition) == (length(problems), length(problems[1].initial_condition)) "The initial condition must be a matrix with the same number of rows as the number of problems."
        @assert all(i -> problems[i].conditions.neumann_edges == conditions[i].neumann_edges, 1:length(problems)) "The Neumann edges in the `i`th `SimpleConditions` must match those from the `i`th problem."
        @assert all(i -> problems[i].conditions.constrained_edges == conditions[i].constrained_edges, 1:length(problems)) "The constrained edges in the `i`th `SimpleConditions` must match those from the `i`th problem."
        @assert all(i -> problems[i].conditions.dirichlet_nodes == conditions[i].dirichlet_nodes, 1:length(problems)) "The Dirichlet nodes in the `i`th `SimpleConditions` must match those from the `i`th problem."
        @assert all(i -> problems[i].conditions.dudt_nodes == conditions[i].dudt_nodes, 1:length(problems)) "The dudt nodes in the `i`th `SimpleConditions` must match those from the `i`th problem."
        @assert all(i -> problems[i].source_function === source_functions[i].fnc, 1:length(problems)) "The source functions must match those from the problems."
        @assert all(i -> problems[i].source_parameters === source_functions[i].parameters, 1:length(problems)) "The source parameters must match those from the problems."
        @assert all(i -> problems[i].flux_function === flux_functions[i].fnc, 1:length(problems)) "The flux functions must match those from the problems."
        @assert all(i -> problems[i].flux_parameters === flux_functions[i].parameters, 1:length(problems)) "The flux parameters must match those from the problems."
        sys = new{length(problems),FG,P,IC,FT,F,S,FF}(mesh, problems, initial_condition, initial_time, final_time, conditions, num_fncs, functions, source_functions, flux_functions)
        _check_fvmsystem_flux_function(sys)
        return sys
    end
end

const FLUX_ERROR = """
                   The flux function errored when evaluated. Please recheck your specification of the function. 
                   If any of your problems have been defined in terms of a diffusion function D(x, y, t, u, p)
                   rather than a flux function, then you need to instead provide a flux function q(x, y, t, α, β, γ, p),
                   recalling the relationship between the two:

                       q(x, y, t, α, β, γ, p) = -D(x, y, t, α*x + β*y + γ, p) .* (α, β)

                   where p is the same argument for both functions.
                   """
struct InvalidFluxError <: Exception end
function Base.showerror(io::IO, ::InvalidFluxError)
    print(io, FLUX_ERROR)
end

function _check_fvmsystem_flux_function(prob::FVMSystem)
    t0 = prob.initial_time
    T = first(each_solid_triangle(prob.mesh.triangulation))
    i, j, k = indices(T)
    p, q, r = get_point(prob.mesh.triangulation, i, j, k)
    px, py = getxy(p)
    qx, qy = getxy(q)
    rx, ry = getxy(r)
    cx, cy = (px + qx + rx) / 3, (py + qy + ry) / 3
    u = prob.initial_condition
    α, β, γ = get_shape_function_coefficients(prob.mesh.triangle_props[T], T, u, prob)
    try
        eval_flux_function(prob, cx, cy, t0, α, β, γ)
    catch e
        if e isa MethodError
            throw(InvalidFluxError())
        else
            rethrow(e)
        end
    end
end

Base.show(io::IO, ::MIME"text/plain", prob::FVMSystem{N}) where {N} = print(io, "FVMSystem with $N equations and time span ($(prob.initial_time), $(prob.final_time))")

@inline map_fidx(prob::FVMSystem, fidx, var) = fidx + prob.cnum_fncs[var]

@inline get_dudt_fidx(prob::FVMSystem, i, var) = get_dudt_nodes(get_conditions(prob, var))[i]
@inline get_neumann_fidx(prob::FVMSystem, i, j, var) = get_neumann_edges(get_conditions(prob, var))[(i, j)]
@inline get_dirichlet_fidx(prob::FVMSystem, i, var) = get_dirichlet_nodes(get_conditions(prob, var))[i]
@inline get_constrained_fidx(prob::FVMSystem, i, j, var) = get_constrained_edges(get_conditions(prob, var))[(i, j)]
@inline eval_condition_fnc(prob::FVMSystem, fidx, var, x, y, t, u) = eval_fnc_in_het_tuple(prob.functions, map_fidx(prob, fidx, var), x, y, t, u)
@inline eval_source_fnc(prob::FVMSystem, var, x, y, t, u) = eval_fnc_in_het_tuple(prob.source_functions, var, x, y, t, u)
@inline is_dudt_node(prob::FVMSystem, node, var) = is_dudt_node(get_conditions(prob, var), node)
@inline is_neumann_edge(prob::FVMSystem, i, j, var) = is_neumann_edge(get_conditions(prob, var), i, j)
@inline is_dirichlet_node(prob::FVMSystem, node, var) = is_dirichlet_node(get_conditions(prob, var), node)
@inline is_constrained_edge(prob::FVMSystem, i, j, var) = is_constrained_edge(get_conditions(prob, var), i, j)
@inline has_condition(prob::FVMSystem, node, var) = has_condition(get_conditions(prob, var), node)
@inline has_dirichlet_nodes(prob::FVMSystem, var) = has_dirichlet_nodes(get_conditions(prob, var))
@inline has_dirichlet_nodes(prob::FVMSystem{N}) where {N} = any(i -> has_dirichlet_nodes(prob, i), 1:N)
@inline get_dirichlet_nodes(prob::FVMSystem, var) = get_dirichlet_nodes(get_conditions(prob, var))
@inline eval_flux_function(prob::FVMSystem, x, y, t, α, β, γ) = eval_all_fncs_in_tuple(prob.flux_functions, x, y, t, α, β, γ)

function FVMSystem(probs::Vararg{FVMProblem,N}) where {N}
    N == 0 && error("There must be at least one problem.")
    mesh = probs[1].mesh
    initial_time = probs[1].initial_time
    final_time = probs[1].final_time
    ic₁ = probs[1].initial_condition
    initial_condition = similar(ic₁, N, length(ic₁))
    for (i, prob) in enumerate(probs)
        initial_condition[i, :] .= prob.initial_condition
    end
    conditions, num_fncs, fncs, source_functions, flux_functions = merge_problem_conditions(probs)
    return FVMSystem(mesh, probs, initial_condition, initial_time, final_time, conditions, num_fncs, fncs, source_functions, flux_functions)
end

function merge_problem_conditions(probs)
    N = length(probs)
    num_fncs = ntuple(N) do i
        length(probs[i].conditions.functions)
    end
    num_fncs_tail = Base.front(num_fncs)
    cnum_fncs = (0, cumsum(num_fncs_tail)...) # 0 sso that we can do cnum_fncs[i] instead of cnum_fncs[i-1], which makes indexing annoying
    fncs = ntuple(N) do i
        probs[i].conditions.functions
    end |> flatten_tuples
    conditions = ntuple(N) do i
        prob = probs[i]
        conds = prob.conditions
        neumann_edges = get_neumann_edges(conds)
        constrained_edges = get_constrained_edges(conds)
        dirichlet_nodes = get_dirichlet_nodes(conds)
        dudt_nodes = get_dudt_nodes(conds)
        return SimpleConditions(neumann_edges, constrained_edges, dirichlet_nodes, dudt_nodes)
    end
    source_functions = ntuple(N) do i
        probs[i].source_function
    end
    source_parameters = ntuple(N) do i
        probs[i].source_parameters
    end
    flux_functions = ntuple(N) do i
        probs[i].flux_function
    end
    flux_parameters = ntuple(N) do i
        probs[i].flux_parameters
    end
    wrapped_source_functions = wrap_functions(source_functions, source_parameters)
    wrapped_flux_functions = wrap_functions(flux_functions, flux_parameters)
    return conditions, cnum_fncs, fncs, wrapped_source_functions, wrapped_flux_functions
end

get_equation(system::FVMSystem, var) = system.problems[var]
get_conditions(system::FVMSystem, var) = system.conditions[var]

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

_neqs(::FVMProblem) = 0 # We test for N > 0 in _rewrap_conditions, so let this be 0
_neqs(::FVMSystem{N}) where {N} = N
_neqs(prob::SteadyFVMProblem) = _neqs(prob.problem)
is_system(prob::AbstractFVMProblem) = _neqs(prob) > 0

"""
    compute_flux(prob::AbstractFVMProblem, i, j, u, t)

Given an edge with indices `(i, j)`, a solution vector `u`, a current time `t`, 
and a problem `prob`, computes the flux `∇⋅q(x, y, t, α, β, γ, p) ⋅ n`, 
where `n` is the normal vector to the edge, `q` is the flux function from `prob`,
`(x, y)` is the midpoint of the edge, `(α, β, γ)` are the shape function coefficients, 
and `p` are the flux parameters from `prob`. If `prob` is an [`FVMSystem`](@ref), the returned 
value is a `Tuple` for each individual flux. The normal vector `n` is a clockwise rotation of 
the edge, meaning pointing right of the edge `(i, j)`.
"""
function compute_flux(prob::AbstractFVMProblem, i, j, u, t)
    tri = prob.mesh.triangulation
    p, q = get_point(tri, i, j)
    px, py = getxy(p)
    qx, qy = getxy(q)
    ex, ey = qx - px, qy - py
    ℓ = norm((ex, ey))
    nx, ny = ey / ℓ, -ex / ℓ
    k = get_adjacent(tri, j, i) # want the vertex in the direction of the normal
    if DelaunayTriangulation.is_boundary_index(k)
        k = get_adjacent(tri, i, j)
    else
        i, j = j, i
    end
    T, props = _safe_get_triangle_props(prob, (i, j, k))
    α, β, γ = get_shape_function_coefficients(props, T, u, prob)
    mx, my = (px + qx) / 2, (py + qy) / 2
    qv = eval_flux_function(prob, mx, my, t, α, β, γ)
    if is_system(prob)
        qn = ntuple(_neqs(prob)) do var
            local qvx, qvy
            qvx, qvy = getxy(qv[var])
            return nx * qvx + ny * qvy
        end
        return qn
    else
        qvx, qvy = getxy(qv)
        return nx * qvx + ny * qvy
    end
end
