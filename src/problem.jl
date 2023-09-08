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
@inline get_triangle_props(prob::AbstractFVMProblem, i, j, k) = prob.mesh.triangle_props[(i, j, k)]
@inline DelaunayTriangulation.get_point(prob::AbstractFVMProblem, i) = get_point(prob.mesh.triangulation, i)
@inline get_volume(prob::AbstractFVMProblem, i) = prob.mesh.cv_volumes[i]
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
    nv = num_points(prob.mesh.triangulation)
    t0 = prob.initial_time
    tf = prob.final_time
    print(io, "FVMProblem with $(nv) nodes and time span ($t0, $tf)")
end
@inline eval_flux_function(prob::FVMProblem, x, y, t, α, β, γ) = prob.flux_function(x, y, t, α, β, γ, prob.flux_parameters)

function FVMProblem(mesh::FVMGeometry, boundary_conditions::BoundaryConditions, internal_conditions::InternalConditions=InternalConditions();
    diffusion_function=nothing,
    diffusion_parameters=nothing,
    source_function=(x, y, t, u, p) -> zero(u),
    source_parameters=nothing,
    flux_function=nothing,
    flux_parameters=nothing,
    initial_condition,
    initial_time=0.0,
    final_time)
    updated_flux_fnc = construct_flux_function(flux_function, diffusion_function, diffusion_parameters)
    conditions = Conditions(mesh, boundary_conditions, internal_conditions)
    return FVMProblem(mesh, conditions,
        updated_flux_fnc, flux_parameters,
        source_function, source_parameters,
        initial_condition, initial_time, final_time)
end

"""
    SteadyFVMProblem(prob::AbstractFVMProblem)

This is a wrapper for an `AbstractFVMProblem` that indicates that the problem is to be solved as a steady-state problem. 
You can then solve the problem using [`solve(::SteadyFVMProblem, ::Any; kwargs...)`](@ref) from NonlinearSolve.jl. Note that the final time is treated 
as infinity rather than as `prob.final_time`.

See also [`FVMProblem`](@ref) and [`FVMSystem`](@ref).
"""
struct SteadyFVMProblem{P<:AbstractFVMProblem,M<:FVMGeometry} <: AbstractFVMProblem
    problem::P
    mesh::M
end
function SteadyFVMProblem(prob::P) where {P}
    return SteadyFVMProblem{P,typeof(prob.mesh)}(prob, prob.mesh)
end
#ConstructionBase.constructorof(::Type{SteadyFVMProblem{P}}) where {P<:FVMProblem} = problem -> begin
#    SteadyFVMProblem{P,typeof(problem.mesh)}(problem)
#end

function Base.show(io::IO, ::MIME"text/plain", prob::SteadyFVMProblem)
    nv = num_points(prob.mesh.triangulation)
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

the same form is used, except `α`, `β`, `γ` are all `Tuple`s so that `α[i]*x + β[i]*y + γ` is the 
approximation to `uᵢ`.

This problem is solved in the same way as a [`FVMProblem`](@ref), except the problem is defined such that 
the solution returns a matrix at each time, where the `(j, i)`th component corresponds to the solution at the `i`th 
node for the `j`th component.

See also [`FVMProblem`](@ref) and [`SteadyFVMProblem`](@ref).

!!! note 
    To construct a steady-state problem for an `FVMSystem`, you need to apply 
    [`SteadyFVMProblem`](@ref) to the system rather than first applying it to
    each individual [`FVMProblem`](@ref) in the system.
"""
struct FVMSystem{N,FG,P,IC,FT} <: AbstractFVMProblem
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
        @assert size(initial_condition) == (length(problems), length(problems[1].initial_condition)) "The initial condition must be a matrix with the same number of rows as the number of problems."
        return new{length(problems),FG,P,IC,FT}(mesh, problems, initial_condition, initial_time, final_time)
    end
end
Base.show(io::IO, ::MIME"text/plain", prob::FVMSystem{N}) where {N} = print(io, "FVMSystem with $N equations and time span ($(prob.initial_time), $(prob.final_time))")

@inline get_dudt_fidx(prob::FVMSystem, i, var) = get_dudt_fidx(get_equation(prob, var), i)
@inline get_neumann_fidx(prob::FVMSystem, i, j, var) = get_neumann_fidx(get_equation(prob, var), i, j)
@inline get_dirichlet_fidx(prob::FVMSystem, i, var) = get_dirichlet_fidx(get_equation(prob, var), i)
@inline get_constrained_fidx(prob::FVMSystem, i, j, var) = get_constrained_fidx(get_equation(prob, var), i, j)
@inline eval_condition_fnc(prob::FVMSystem, fidx, var, x, y, t, u) = eval_condition_fnc(get_equation(prob, var), fidx, x, y, t, u)
@inline eval_source_fnc(prob::FVMSystem, var, x, y, t, u) = eval_source_fnc(get_equation(prob, var), x, y, t, u)
@inline is_dudt_node(prob::FVMSystem, node, var) = is_dudt_node(get_equation(prob, var), node)
@inline is_neumann_edge(prob::FVMSystem, i, j, var) = is_neumann_edge(get_equation(prob, var), i, j)
@inline is_dirichlet_node(prob::FVMSystem, node, var) = is_dirichlet_node(get_equation(prob, var), node)
@inline is_constrained_edge(prob::FVMSystem, i, j, var) = is_constrained_edge(get_equation(prob, var), i, j)
@inline has_condition(prob::FVMSystem, node, var) = has_condition(get_equation(prob, var), node)
@inline has_dirichlet_nodes(prob::FVMSystem{N}) where {N} = any(i -> has_dirichlet_nodes(get_equation(prob, i)), 1:N)
@inline get_dirichlet_nodes(prob::FVMSystem, var) = get_dirichlet_nodes(get_equation(prob, var))
@inline eval_flux_function(prob::FVMSystem{N}, x, y, t, α, β, γ) where {N} = ntuple(i -> eval_flux_function(get_equation(prob, i), x, y, t, α[i], β[i], γ[i]), Val(N))

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
    return FVMSystem(mesh, probs, initial_condition, initial_time, final_time)
end

get_equation(system::FVMSystem, var) = system.problems[var]

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
    i, j, k = indices(DelaunayTriangulation.contains_triangle(tri, i, j, k)[1]) # so that the key in triangle_props is (i, j, k) not e.g. (j, k, i)
    α, β, γ = get_shape_function_coefficients(prob.mesh.triangle_props[(i, j, k)], (i, j, k), u, prob)
    mx, my = (px + qx) / 2, (py + qy) / 2
    qv = eval_flux_function(prob, mx, my, t, α, β, γ)
    if is_system(prob)
        qn = ntuple(Val(_neqs(prob))) do var
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
