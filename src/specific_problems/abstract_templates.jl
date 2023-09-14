"""
    abstract type AbstractFVMTemplate <: AbstractFVMProblem

An abstract type that defines some specific problems. These problems are those that could 
be defined directly using [`FVMProblem`](@ref)s, but are common enough
that (1) it is useful to have them defined here, and (2) it is
useful to have them defined in a way that is more efficient than
with a default implementation (e.g. exploiting linearity). The 
problems are all defined as subtypes of a common abstract type, 
namely, `AbstractFVMTemplate` (the home of this docstring), which itself is a subtype of
`AbstractFVMProblem`. 

To understand how to make use of these specific problems, either 
see the docstring for each problem, or see the 
"Solvers for Specific Problems, and Writing Your Own" section of the docs.

To see the full list of problems, do 
```julia-repl
julia> using FiniteVolumeMethod

julia> subtypes(FiniteVolumeMethod.AbstractFVMTemplate)
5-element Vector{Any}:
 DiffusionEquation
 LaplacesEquation
 LinearReactionDiffusionEquation
 MeanExitTimeProblem
 PoissonsEquation
```

The constructor for each problem is defined in its docstring. Note that all the problems above are exported.

These problems can all be solved using the standard `solve` interface from 
DifferentialEquations.jl, just like for [`FVMProblem`](@ref)s. The only exception 
is for steady state problems, in which case the `solve` interface is still used, except 
the interface is from LinearSolve.jl.
"""
abstract type AbstractFVMTemplate <: AbstractFVMProblem end

include("diffusion_equation.jl")
include("linear_reaction_diffusion_equations.jl")
include("mean_exit_time.jl")
include("poissons_equation.jl")
include("laplaces_equation.jl")

export DiffusionEquation
export LinearReactionDiffusionEquation
export MeanExitTimeProblem
export PoissonsEquation
export LaplacesEquation

"""
    solve(prob::AbstractFVMTemplate, args...; kwargs...)

Solve the problem `prob` using the standard `solve` interface from DifferentialEquations.jl. For 
steady state problems, the interface is from LinearSolve.jl.
"""
CommonSolve.solve(prob::AbstractFVMTemplate, args...; kwargs...) = CommonSolve.solve(prob.problem, args...; kwargs...)

@doc raw"""
    triangle_contributions!(A, mesh, conditions, diffusion_function, diffusion_parameters)
    
Add the contributions from each triangle to the matrix `A`, based on the equation 

```math 
\dv{u_i}{t} = \frac{1}{V_i}\sum_{\sigma \in \mathcal E_i} D(\vb x_\sigma)\left[\left(s_{k, 11}n_\sigma^x + s_{k, 21}n_\sigma^y\right)u_{k1} + \left(s_{k, 12}n_\sigma^x + s_{k, 22}n_\sigma^y\right)u_{k2} + \left(s_{k, 13}n_\sigma^x + s_{k, 23}n_\sigma^y\right)u_{k3}\right]L_\sigma + S_i, 
```
as explained in the docs. Will not update any rows corresponding to 
[`Dirichlet`](@ref) or [`Dudt`](@ref) nodes.
"""
function triangle_contributions!(A, mesh, conditions, diffusion_function, diffusion_parameters)
    for T in each_solid_triangle(mesh.triangulation)
        ijk = indices(T)
        i, j, k = ijk
        props = get_triangle_props(mesh, i, j, k)
        s₁₁, s₁₂, s₁₃, s₂₁, s₂₂, s₂₃, s₃₁, s₃₂, s₃₃ = props.shape_function_coefficients
        for (edge_index, (e1, e2)) in enumerate(((i, j), (j, k), (k, i)))
            x, y, nx, ny, ℓ = get_cv_components(props, edge_index)
            D = diffusion_function(x, y, diffusion_parameters)
            Dℓ = D * ℓ
            a123 = (Dℓ * (s₁₁ * nx + s₂₁ * ny),
                Dℓ * (s₁₂ * nx + s₂₂ * ny),
                Dℓ * (s₁₃ * nx + s₂₃ * ny))
            e1_hascond = has_condition(conditions, e1)
            e2_hascond = has_condition(conditions, e2)
            for vert in 1:3
                e1_hascond || (A[e1, ijk[vert]] += a123[vert] / get_volume(mesh, e1))
                e2_hascond || (A[e2, ijk[vert]] -= a123[vert] / get_volume(mesh, e2))
            end
        end
    end
end

@doc raw"""
    apply_dirichlet_conditions!(initial_condition, mesh, conditions)

Applies the Dirichlet conditions specified in `conditions` to the `initial_condition`. The boundary 
conditions are assumed to take the form `a(x, y, t, u, p) -> Number`, but `t` and `u` are passed 
as `nothing`. Note that this assumes that the associated system `(A, b)` is such that `A[i, :]` is all 
zero, and `b[i]` is zero, where `i` is a node with a Dirichlet condition.
"""
function apply_dirichlet_conditions!(initial_condition, mesh, conditions)
    for (i, function_index) in get_dirichlet_nodes(conditions)
        x, y = get_point(mesh, i)
        initial_condition[i] = eval_condition_fnc(conditions, function_index, x, y, nothing, nothing)
    end
end

@doc raw"""
    apply_dudt_conditions!(b, mesh, conditions)

Applies the Dudt conditions specified in `conditions` to the `b` vector. The boundary   
conditions are assumed to take the form `a(x, y, t, u, p) -> Number`, but `t` and `u` are passed
as `nothing`. Note that this assumes that the associated system `(A, b)` is such that `A[i, :]` is all
zero, so that replacing `b[i]` with the boundary condition will set `duᵢ/dt = b[i]`.
"""
function apply_dudt_conditions!(b, mesh, conditions)
    for (i, function_index) in get_dudt_nodes(conditions)
        if !is_dirichlet_node(conditions, i) # overlapping edges can be both Dudt and Dirichlet. Dirichlet takes precedence
            x, y = get_point(mesh, i)
            b[i] = eval_condition_fnc(conditions, function_index, x, y, nothing, nothing)
        end
    end
end

@doc raw"""
    boundary_edge_contributions!(A, b, mesh, conditions, diffusion_function, diffusion_parameters)

Add the contributions from each boundary edge to the matrix `A`, based on the equation 

```math 
\dv{u_i}{t} = \frac{1}{V_i}\sum_{\sigma \in \mathcal E_i} D(\vb x_\sigma)\left[\left(s_{k, 11}n_\sigma^x + s_{k, 21}n_\sigma^y\right)u_{k1} + \left(s_{k, 12}n_\sigma^x + s_{k, 22}n_\sigma^y\right)u_{k2} + \left(s_{k, 13}n_\sigma^x + s_{k, 23}n_\sigma^y\right)u_{k3}\right]L_\sigma + S_i, 
```

as explained in the docs. Will not update any rows corresponding to 
[`Dirichlet`](@ref) or [`Dudt`](@ref) nodes.
"""
function boundary_edge_contributions!(A, b, mesh, conditions,
    diffusion_function, diffusion_parameters)
    non_neumann_boundary_edge_contributions!(A, mesh, conditions, diffusion_function, diffusion_parameters)
    neumann_boundary_edge_contributions!(b, mesh, conditions, diffusion_function, diffusion_parameters)
    return nothing
end

@doc raw"""
    neumann_boundary_edge_contributions!(b, mesh, conditions, diffusion_function, diffusion_parameters)

Add the contributions from each Neumann boundary edge to the vector `b`, based on the equation

```math 
\dv{u_i}{t} = \frac{1}{V_i}\sum_{\sigma \in \mathcal E_i} D(\vb x_\sigma)\left[\grad u(\vb x_\sigma) \vdot \vu n\right]L_\sigma + S_i,
```

as explained in the docs. Will not update any rows corresponding to 
[`Dirichlet`](@ref) or [`Dudt`](@ref) nodes. This function will pass `nothing` in place 
of the arguments `u` and `t` in the boundary condition functions.
"""
function neumann_boundary_edge_contributions!(b, mesh, conditions, diffusion_function, diffusion_parameters)
    for (e, fidx) in get_neumann_edges(conditions)
        i, j = DelaunayTriangulation.edge_indices(e)
        _, _, mᵢx, mᵢy, mⱼx, mⱼy, ℓ, _, _ = get_boundary_cv_components(mesh, i, j)
        Dᵢ = diffusion_function(mᵢx, mᵢy, diffusion_parameters)
        Dⱼ = diffusion_function(mⱼx, mⱼy, diffusion_parameters)
        i_hascond = has_condition(conditions, i)
        j_hascond = has_condition(conditions, j)
        aᵢ = eval_condition_fnc(conditions, fidx, mᵢx, mᵢy, nothing, nothing)
        aⱼ = eval_condition_fnc(conditions, fidx, mⱼx, mⱼy, nothing, nothing)
        i_hascond || (b[i] += Dᵢ * aᵢ * ℓ / get_volume(mesh, i))
        j_hascond || (b[j] += Dⱼ * aⱼ * ℓ / get_volume(mesh, j))
    end
    return nothing
end

@doc raw"""
    neumann_boundary_edge_contributions!(F, mesh, conditions, diffusion_function, diffusion_parameters, u, t)

Add the contributions from each Neumann boundary edge to the vector `F`, based on the equation

```math 
\dv{u_i}{t} = \frac{1}{V_i}\sum_{\sigma \in \mathcal E_i} D(\vb x_\sigma)\left[\grad u(\vb x_\sigma) \vdot \vu n\right]L_\sigma + S_i,
```

as explained in the docs. Will not update any rows corresponding to 
[`Dirichlet`](@ref) or [`Dudt`](@ref) nodes. 
"""
function neumann_boundary_edge_contributions!(F, mesh, conditions, diffusion_function, diffusion_parameters, u, t)
    for (e, fidx) in FVM.get_neumann_edges(conditions)
        i, j = DelaunayTriangulation.edge_indices(e)
        _, _, mᵢx, mᵢy, mⱼx, mⱼy, ℓ, _, _ = FVM.get_boundary_cv_components(mesh, i, j)
        Dᵢ = diffusion_function(mᵢx, mᵢy, diffusion_parameters)
        Dⱼ = diffusion_function(mⱼx, mⱼy, diffusion_parameters)
        i_hascond = FVM.has_condition(conditions, i)
        j_hascond = FVM.has_condition(conditions, j)
        uᵢ_itp = two_point_interpolant(mesh, u, i, j, mᵢx, mᵢy)
        uⱼ_itp = two_point_interpolant(mesh, u, i, j, mⱼx, mⱼy)
        aᵢ = FVM.eval_condition_fnc(conditions, fidx, mᵢx, mᵢy, t, uᵢ_itp)
        aⱼ = FVM.eval_condition_fnc(conditions, fidx, mⱼx, mⱼy, t, uⱼ_itp)
        i_hascond || (F[i] += Dᵢ * aᵢ * ℓ / FVM.get_volume(mesh, i))
        j_hascond || (F[j] += Dⱼ * aⱼ * ℓ / FVM.get_volume(mesh, j))
    end
    return nothing
end

@doc raw"""
    non_neumann_boundary_edge_contributions!(A, mesh, conditions, diffusion_function, diffusion_parameters)

Add the contributions from each non-Neumann boundary edge to the matrix `A`, based on the equation

```math
\dv{u_i}{t} = \frac{1}{V_i}\sum_{\sigma \in \mathcal E_i} D(\vb x_\sigma)\left[\left(s_{k, 11}n_\sigma^x + s_{k, 21}n_\sigma^y\right)u_{k1} + \left(s_{k, 12}n_\sigma^x + s_{k, 22}n_\sigma^y\right)u_{k2} + \left(s_{k, 13}n_\sigma^x + s_{k, 23}n_\sigma^y\right)u_{k3}\right]L_\sigma + S_i, 
```

as explained in the docs. Will not update any rows corresponding to 
[`Dirichlet`](@ref) or [`Dudt`](@ref) nodes.
"""
function non_neumann_boundary_edge_contributions!(A, mesh, conditions, diffusion_function, diffusion_parameters)
    for e in keys(get_boundary_edge_map(mesh.triangulation))
        i, j = DelaunayTriangulation.edge_indices(e)
        if !is_neumann_edge(conditions, i, j)
            nx, ny, mᵢx, mᵢy, mⱼx, mⱼy, ℓ, T, props = get_boundary_cv_components(mesh, i, j)
            ijk = indices(T)
            s₁₁, s₁₂, s₁₃, s₂₁, s₂₂, s₂₃, s₃₁, s₃₂, s₃₃ = props.shape_function_coefficients
            Dᵢ = diffusion_function(mᵢx, mᵢy, diffusion_parameters)
            Dⱼ = diffusion_function(mⱼx, mⱼy, diffusion_parameters)
            i_hascond = has_condition(conditions, i)
            j_hascond = has_condition(conditions, j)
            aᵢ123 = (Dᵢ * ℓ * (s₁₁ * nx + s₂₁ * ny),
                Dᵢ * ℓ * (s₁₂ * nx + s₂₂ * ny),
                Dᵢ * ℓ * (s₁₃ * nx + s₂₃ * ny))
            aⱼ123 = (Dⱼ * ℓ * (s₁₁ * nx + s₂₁ * ny),
                Dⱼ * ℓ * (s₁₂ * nx + s₂₂ * ny),
                Dⱼ * ℓ * (s₁₃ * nx + s₂₃ * ny))
            for vert in 1:3
                i_hascond || (A[i, ijk[vert]] += aᵢ123[vert] / get_volume(mesh, i))
                j_hascond || (A[j, ijk[vert]] += aⱼ123[vert] / get_volume(mesh, i))
            end
        end
    end
    return nothing
end

"""
    create_rhs_b(mesh, conditions, source_function, source_parameters)

Create the vector `b` defined by 

    b = [source_function(x, y, source_parameters) for (x, y) in each_point(mesh.triangulation)],

and `b[i] = 0` whenever `i` is a Dirichlet node.
"""
function create_rhs_b(mesh, conditions, source_function, source_parameters)
    b = zeros(DelaunayTriangulation.num_solid_vertices(mesh.triangulation))
    for i in each_solid_vertex(mesh.triangulation)
        if !is_dirichlet_node(conditions, i)
            p = get_point(mesh, i)
            x, y = getxy(p)
            b[i] = source_function(x, y, source_parameters)
        end
    end
    return b
end

@doc raw"""
    apply_steady_dirichlet_conditions!(A, b, mesh, conditions)

Applies the Dirichlet conditions specified in `conditions` to the `initial_condition`. The boundary 
conditions are assumed to take the form `a(x, y, t, u, p) -> Number`, but `t` and `u` are passed 
as `nothing`. Note that this assumes that the associated system `(A, b)` is such that `A[i, :]` is all 
zero, and `b[i]` is zero, where `i` is a node with a Dirichlet condition.

For a steady problem `Au = b`, applies the Dirichlet boundary conditions specified by `conditions` 
so that `A[i, i] = 1` and `b[i]` is the condition, where `i` is a boundary node. Note that this 
assumes that all of `A[i, :]` is zero before setting `A[i, i] = 1`.
"""
function apply_steady_dirichlet_conditions!(A, b, mesh, conditions)
    for (i, function_index) in get_dirichlet_nodes(conditions)
        x, y = get_point(mesh, i)
        b[i] = eval_condition_fnc(conditions, function_index, x, y, nothing, nothing)
        A[i, i] = 1.0
    end
end