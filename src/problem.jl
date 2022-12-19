"""
    FVMProblem{iip_flux,FG,BC,F,FP,R,RP,IC,FT}

Struct representing the PDE problem. 

# Fields 
- `mesh::FG`: The underlying mesh, given as a [`FVMGeometry`](@ref) struct. 
- `boundary_conditions::BC`: The boundary conditions, given a a [`BoundaryConditions`](@ref) struct. 
- `flux_function::F`: The function for the flux vector, taking either of the forms `flux!(q, x, y, t, α, β, γ, p)` (if `iip_flux`) and `flux(x, y, t, α, β, γ, p)` (if `!iip_flux`), where `u = αx + βy + γ`.
- `flux_parameters::FP`: The argument `p` in `flux_function`.
- `reaction_function::R`: The function for the reaction term, taking the form `R(x, y, t, u, p)`.
- `reaction_parameters::RP`: The argument `p` in `reaction_function`.
- `initial_condition::IC`: The initial condition for the problem, with `initial_condition[i]` the initial value at the `i`th node of the mesh. 
- `initial_time::FT`: The time to start solving the PDE at. 
- `final_time::FT`: The time to stop solving the PDE at. 
- `steady::Bool`: Whether `∂u/∂t = 0` or not. Not currently used; only non-steady problems are currently supported (see https://github.com/DanielVandH/FiniteVolumeMethod.jl/issues/16).
"""
struct FVMProblem{iip_flux,FG,BC,F,FP,R,RP,IC,FT}
    mesh::FG
    boundary_conditions::BC
    flux_function::F
    flux_parameters::FP
    reaction_function::R
    reaction_parameters::RP
    initial_condition::IC
    initial_time::FT
    final_time::FT
    steady::Bool
end

"""
    FVMProblem(mesh, boundary_conditions;
        iip_flux=true,
        diffusion_function=nothing,
        diffusion_parameters=nothing,
        reaction_function=nothing,
        reaction_parameters=nothing,
        delay_function=nothing,
        delay_parameters=nothing,
        flux_function=nothing,
        flux_parameters=nothing,
        initial_condition,
        initial_time=0.0,
        final_time,
        steady=false,
        q_storage=SVector{2,Float64})

Constructor for the [`FVMProblem`](@ref).

# Arguments 
- `mesh`: The [`FVMGeometry`](@ref) representing the underlying mesh. 
- `boundary_conditions`: The [`BoundaryConditions`](@ref) representing the boundary conditions for the problem. 

# Keyword Arguments 
- `iip_flux=true`: Whether the flux vector is computed in-place or not.
- `diffusion_function=nothing`: If `flux_function===nothing`, this can be used to provide a diffusion term for the reaction-diffusion formulation, taking the form `D(x, y, t, u, p)`. See also [`construct_flux_function`](@ref).
- `diffusion_parameters=nothing`: The argument `p` in the diffusion function. 
- `reaction_function=nothing`: The reaction term, taking the form `R(x, y, t, u, p)`. If not provided, it is set to the zero function. See also [`construct_reaction_function`](@ref).
- `reaction_parameters=nothing`: The argument `p` in the reaction function. 
- `delay_function=nothing`: The delay function `T(x, y, t, u, p)` for the PDE that can be used to scale the diffusion and reaction functions, assuming `flux_function===nothing`. See also [`construct_reaction_function`](@ref) and [`construct_flux_function`](@ref).
- `delay_parameters=nothing`: The argument `p` in the delay function. 
- `flux_function=nothing`: The flux function, taking either of the forms `flux!(q, x, y, t, α, β, γ, p)` (if `iip_flux`) and `flux(x, y, t, α, β, γ, p)` (if `!iip_flux`), where `u = αx + βy + γ`. If `flux_function===nothing`, thne this function is constructed from the diffusion and delay functions. 
- `flux_parameters=nothing`: The argument `p` in the flux function. 
- `initial_condition`: The initial condition for the problem, with `initial_condition[i]` the initial value at the `i`th node of the mesh. 
- `initial_time=0.0`: The time to start solving the PDE at. 
- `final_time`: The time to stop solving the PDE at. 
- `steady::Bool`: Whether `∂u/∂t = 0` or not. Not currently used; only non-steady problems are currently supported.

# Outputs 
Returns the [`FVMProblem`](@ref) object.
"""
function FVMProblem(mesh, boundary_conditions;
    iip_flux=true,
    diffusion_function=nothing,
    diffusion_parameters=nothing,
    reaction_function=nothing,
    reaction_parameters=nothing,
    delay_function=nothing,
    delay_parameters=nothing,
    flux_function=nothing,
    flux_parameters=nothing,
    initial_condition,
    initial_time=0.0,
    final_time,
    steady=false)
    updated_flux_fnc = construct_flux_function(iip_flux, flux_function,
        delay_function, delay_parameters,
        diffusion_function, diffusion_parameters)
    updated_reaction_fnc = construct_reaction_function(reaction_function, reaction_parameters,
        delay_function, delay_parameters)
    return FVMProblem{iip_flux,
        typeof(mesh),typeof(boundary_conditions),
        typeof(updated_flux_fnc),typeof(flux_parameters),
        typeof(updated_reaction_fnc),typeof(reaction_parameters),
        typeof(initial_condition),
        typeof(initial_time)}(
        mesh, boundary_conditions,
        updated_flux_fnc, flux_parameters,
        updated_reaction_fnc, reaction_parameters,
        initial_condition,
        initial_time, final_time,
        steady
    )
end
SciMLBase.isinplace(::FVMProblem{iip_flux,FG,BC,F,FP,R,RP,IC,FT}) where {iip_flux,FG,BC,F,FP,R,RP,IC,FT} = iip_flux
get_boundary_conditions(prob::FVMProblem) = prob.boundary_conditions
get_initial_condition(prob::FVMProblem) = prob.initial_condition
get_initial_time(prob::FVMProblem) = prob.initial_time
get_final_time(prob::FVMProblem) = prob.final_time
get_time_span(prob::FVMProblem) = (get_initial_time(prob), get_final_time(prob))

"""
    construct_flux_function(iip_flux,
        flux_function,
        delay_function, delay_parameters,
        diffusion_function, diffusion_parameters)

Constructs the flux function. The arguments are as in [`FVMProblem`](@ref), and the output depends on the following, where `D`
denotes the diffusion function, `T` the delay function, `dₚ` the diffusion parameters, and `tₚ` the delay parameters: 

- If `flux_function===nothing` and `delay_function===nothing`, defines the flux function as `(x, y, t, α, β, γ, _) -> -D(x, y, t, αx + βy + γ, dₚ)[α, β]`.
- If `flux_function===nothing`, defines the flux function as `(x, y, t, α, β, γ, _) -> -T(x, y, t, αx + βy + γ, tₚ)D(x, y, t, αx + βy + γ, dₚ)[α, β]`.

If `iip_flux`, then the functions above have an extra argument in the first position, `q`, that the flux vectors are stored in. Otherwise, the flux vector is 
returned as a tuple `(q1, q2)`. If `flux_function !== nothing`, then just returns `flux_function`.
"""
function construct_flux_function(iip_flux,
    flux_function,
    delay_function, delay_parameters,
    diffusion_function, diffusion_parameters) 
    if isnothing(flux_function)
        if !isnothing(delay_function)
            if iip_flux
                flux_fnc = (q, x, y, t, α, β, γ, p) -> begin
                    u = α * x + β * y + γ
                    q[1] = -delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * α
                    q[2] = -delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * β
                    return nothing
                end
                return flux_fnc
            else
                flux_fnc = (x, y, t, α, β, γ, p) -> begin
                    u = α * x + β * y + γ
                    q1 = -delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * α
                    q2 = -delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * β
                    return q1, q2
                end
                return flux_fnc
            end
            return flux_fnc
        else
            if iip_flux
                flux_fnc = (q, x, y, t, α, β, γ, p) -> begin
                    u = α * x + β * y + γ
                    q[1] = -diffusion_function(x, y, t, u, diffusion_parameters) * α
                    q[2] = -diffusion_function(x, y, t, u, diffusion_parameters) * β
                    return nothing
                end
                return flux_fnc
            else
                flux_fnc = (x, y, t, α, β, γ, p) -> begin
                    u = α * x + β * y + γ
                    q1 = -diffusion_function(x, y, t, u, diffusion_parameters) * α
                    q2 = -diffusion_function(x, y, t, u, diffusion_parameters) * β
                    return q1, q2
                end
                return flux_fnc
            end
            return flux_fnc
        end
    else
        return flux_function
    end
end

"""
    construct_reaction_function(reaction_function, reaction_parameters,
        delay_function, delay_parameters)

Constructs the reaction function. The arguments are as in [`FVMProblem`](@ref), and the output depends on the following, where `R`
            denotes the reaction function, `T` the delay function, `rₚ` the reaction parameters, and `tₚ` the delay parameters: 

- If `reaction_function===nothing`, defines the reaction function as `(x, y, t, u, _) = 0.0`.
- If `reaction_function !== nothing` and `delay_function !== nothing`, defines the reaction function as `(x, y, t, u, _) -> T(x, y, t, u, tₚ)R(x, y, t, u, rₚ)`.
- Otherwise, just returns `reaction_function`.
"""
function construct_reaction_function(reaction_function, reaction_parameters,
    delay_function, delay_parameters)
    if !isnothing(reaction_function)
        if !isnothing(delay_function)
            reaction_fnc = (x, y, t, u, p) -> begin
                return delay_function(x, y, t, u, delay_parameters) * reaction_function(x, y, t, u, reaction_parameters)
            end
            return reaction_fnc
        else
            return reaction_function
        end
    else
        reaction_fnc = ((x, y, t, u::T, p) where {T}) -> zero(T)
        return reaction_fnc
    end
end

@inline get_mesh(prob::FVMProblem) = prob.mesh
@inline gets(prob::FVMProblem, T) = gets(get_mesh(prob), T)
@inline gets(prob::FVMProblem, T, i) = gets(prob, T)[i]
@inline get_midpoints(prob::FVMProblem, T) = get_midpoints(get_mesh(prob), T)
@inline get_midpoints(prob::FVMProblem, T, i) = get_midpoints(prob, T)[i]
@inline get_control_volume_edge_midpoints(prob::FVMProblem, T) = get_control_volume_edge_midpoints(get_mesh(prob), T)
@inline get_control_volume_edge_midpoints(prob::FVMProblem, T, i) = get_control_volume_edge_midpoints(prob, T)[i]
@inline get_normals(prob::FVMProblem, T) = get_normals(get_mesh(prob), T)
@inline get_normals(prob::FVMProblem, T, i) = get_normals(prob, T)[i]
@inline get_lengths(prob::FVMProblem, T) = get_lengths(get_mesh(prob), T)
@inline get_lengths(prob::FVMProblem, T, i) = get_lengths(prob, T)[i]
@inline get_flux(prob::FVMProblem, x, y, t, α, β, γ) = prob.flux_function(x, y, t, α, β, γ, prob.flux_parameters)
@inline get_flux!(flux_cache, prob::FVMProblem, x, y, t, α, β, γ) = prob.flux_function(flux_cache, x, y, t, α, β, γ, prob.flux_parameters)
@inline get_interior_or_neumann_nodes(prob::FVMProblem) = get_interior_or_neumann_nodes(get_boundary_conditions(prob))
@inline get_interior_elements(prob::FVMProblem) = get_interior_elements(get_mesh(prob))
@inline get_interior_edges(prob::FVMProblem, T) = get_interior_edges(get_mesh(prob), T)
@inline get_boundary_elements(prob::FVMProblem) = get_boundary_elements(get_mesh(prob))
@inline DelaunayTriangulation.get_point(prob::FVMProblem, j) = get_point(get_mesh(prob), j)
@inline get_points(prob::FVMProblem) = get_points(get_mesh(prob))
@inline get_volumes(prob::FVMProblem, j) = get_volumes(get_mesh(prob), j)
@inline get_reaction(prob::FVMProblem, x, y, t, u) = prob.reaction_function(x, y, t, u, prob.reaction_parameters)
@inline get_dirichlet_nodes(prob::FVMProblem) = get_dirichlet_nodes(get_boundary_conditions(prob))
@inline get_neumann_nodes(prob::FVMProblem) = get_neumann_nodes(get_boundary_conditions(prob))
@inline get_dudt_nodes(prob::FVMProblem) = get_dudt_nodes(get_boundary_conditions(prob))
@inline get_boundary_nodes(prob::FVMProblem) = get_boundary_nodes(get_mesh(prob))
@inline get_boundary_function_parameters(prob::FVMProblem, i) = get_boundary_function_parameters(get_boundary_conditions(prob), i)
@inline get_neighbours(prob::FVMProblem) = get_neighbours(get_mesh(prob))
@inline map_node_to_segment(prob::FVMProblem, j) = map_node_to_segment(get_boundary_conditions(prob), j)
@inline is_interior_or_neumann_node(prob::FVMProblem, j) = is_interior_or_neumann_node(get_boundary_conditions(prob), j)
@inline evaluate_boundary_function(prob::FVMProblem, idx, x, y, t, u) = evaluate_boundary_function(get_boundary_conditions(prob), idx, x, y, t, u)
@inline DelaunayTriangulation.num_points(prob::FVMProblem) = DelaunayTriangulation.num_points(get_points(prob))
@inline num_boundary_edges(prob::FVMProblem) = num_boundary_edges(get_mesh(prob)) # This will be the same value as the above, but this makes the code clearer to read
@inline get_adjacent(prob::FVMProblem) = get_adjacent(get_mesh(prob))
@inline get_adjacent2vertex(prob::FVMProblem) = get_adjacent2vertex(get_mesh(prob))
@inline get_elements(prob::FVMProblem) = get_elements(get_mesh(prob))
@inline get_element_type(prob::FVMProblem) = get_element_type(get_mesh(prob))
