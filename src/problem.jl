abstract type AbstractFVMProblem end

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

Constructs an `FVMProblem`. See also [`FVMSystem`](@ref), [`SteadyFVMProblem`](@ref), and 
[`ConstrainedFVMProblem`](@ref).

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
function _rewrap_conditions(prob::FVMProblem, neqs::Val{N}, constrained::Val{B}) where {N,B}
    new_conds = _rewrap_conditions(prob.conditions, eltype(prob.initial_condition), neqs, constrained)
    return FVMProblem(prob.mesh, new_conds, prob.flux_function, prob.flux_parameters, prob.source_function, prob.source_parameters, prob.initial_condition, prob.initial_time, prob.final_time)
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
    SteadyFVMProblem{P<:AbstractFVMProblem}

This is a wrapper for an `AbstractFVMProblem` that indicates that the problem is to be solved as a steady-state problem. 
You can then solve the problem using `solve` from (Simple)NonlinearSolve.jl.

See also [`FVMProblem`](@ref), [`FVMSystem`](@ref), and [`ConstrainedFVMProblem`](@ref).
"""
struct SteadyFVMProblem{P<:AbstractFVMProblem} <: AbstractFVMProblem
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

See also [`FVMProblem`](@ref), [`SteadyFVMProblem`](@ref), and [`ConstrainedFVMProblem`](@ref).
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
function _rewrap_conditions(prob::FVMSystem{N}, constrained::Val{B}) where {N,B}
    problems = ntuple(i -> _rewrap_conditions(prob.problems[i], Val(N), constrained), N)
    return FVMSystem(prob.mesh, problems, prob.initial_condition, prob.initial_time, prob.final_time)
end

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
    wrapped_probs = ntuple(i -> _rewrap_conditions(probs[i], Val(N), Val(false)), N)
    return FVMSystem(mesh, wrapped_probs, initial_condition, initial_time, final_time)
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

_neqs(::FVMProblem) = 0 # We test for N > 0 in _rewrap_conditions, so let this be 0
_neqs(::FVMSystem{N}) where {N} = N
_neqs(prob::SteadyFVMProblem) = _neqs(prob.problem)
"""
    ConstrainedFVMProblem{P<:AbstractFVMProblem} <: AbstractFVMProblem

This is a wrapper for an `AbstractFVMProblem` that indicates that the problem is to be solved as a constrained problem.
You can then solve the problem using `solve` from DifferentialEquations.jl, treating it as a DAE. See the docs for some 
examples.
"""
struct ConstrainedFVMProblem{P<:AbstractFVMProblem} <: AbstractFVMProblem 
    prob::P 
    function ConstrainedFVMProblem(prob::P) where {P}
        wrapped_problem = _rewrap_conditions(prob, Val(_neqs(prob)), Val(true))
        return new{P}(wrapped_problem)
    end
end
