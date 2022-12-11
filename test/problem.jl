## Make sure that the flux function is being constructed correctly
for iip_flux in (true, false)
        flux_function = nothing
        flux_parameters = nothing
        delay_function = nothing
        delay_parameters = nothing
        diffusion_function = (x, y, t, u, p) -> x * y
        diffusion_parameters = nothing
        flux_fnc = FVM.construct_flux_function(iip_flux, flux_function, delay_function, delay_parameters, diffusion_function, diffusion_parameters)
        x, y, t, α, β, γ, p = rand(), rand(), rand(), rand(), rand(), rand(), nothing
        if iip_flux
            q = zeros(2)
            @inferred flux_fnc(q, x, y, t, α, β, γ, p)
            SHOW_WARNTYPE && @code_warntype flux_fnc(q, x, y, t, α, β, γ, p)
            u = α * x + β * y + γ
            @test all(q .≈ [-diffusion_function(x, y, t, u, diffusion_parameters) * α, -diffusion_function(x, y, t, u, diffusion_parameters) * β])
        else
            @inferred flux_fnc(x, y, t, α, β, γ, p)
            SHOW_WARNTYPE && @code_warntype flux_fnc(x, y, t, α, β, γ, p)
            q = flux_fnc(x, y, t, α, β, γ, p)
            u = α * x + β * y + γ
            @test all(q .≈ [-diffusion_function(x, y, t, u, diffusion_parameters) * α, -diffusion_function(x, y, t, u, diffusion_parameters) * β])
        end

        diffusion_function = (x, y, t, u, p) -> x * y + p
        diffusion_parameters = 3.7
        x, y, t, α, β, γ, p = rand(), rand(), rand(), rand(), rand(), rand(), nothing
        flux_fnc = FVM.construct_flux_function(iip_flux, flux_function, delay_function, delay_parameters, diffusion_function, diffusion_parameters)
        if iip_flux
            q = zeros(2)
            @inferred flux_fnc(q, x, y, t, α, β, γ, p)
            SHOW_WARNTYPE && @code_warntype flux_fnc(q, x, y, t, α, β, γ, p)
            u = α * x + β * y + γ
            @test all(q .≈ [-diffusion_function(x, y, t, u, diffusion_parameters) * α, -diffusion_function(x, y, t, u, diffusion_parameters) * β])
        else
            @inferred flux_fnc(x, y, t, α, β, γ, p)
            SHOW_WARNTYPE && @code_warntype flux_fnc(x, y, t, α, β, γ, p)
            q = flux_fnc(x, y, t, α, β, γ, p)
            u = α * x + β * y + γ
            @test all(q .≈ [-diffusion_function(x, y, t, u, diffusion_parameters) * α, -diffusion_function(x, y, t, u, diffusion_parameters) * β])
        end

        diffusion_function = (x, y, t, u, p) -> x * y
        diffusion_parameters = nothing
        delay_function = (x, y, t, u, p) -> 1 / (1 + exp(-t))
        delay_parameters = nothing
        flux_fnc = FVM.construct_flux_function(iip_flux, flux_function, delay_function, delay_parameters, diffusion_function, diffusion_parameters)
        x, y, t, α, β, γ, p = rand(), rand(), rand(), rand(), rand(), rand(), nothing
        if iip_flux
            q = zeros(2)
            @inferred flux_fnc(q, x, y, t, α, β, γ, p)
            SHOW_WARNTYPE && @code_warntype flux_fnc(q, x, y, t, α, β, γ, p)
            u = α * x + β * y + γ
            @test all(q .≈ [-delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * α,
                -delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * β])
        else
            @inferred flux_fnc(x, y, t, α, β, γ, p)
            SHOW_WARNTYPE && @code_warntype flux_fnc(x, y, t, α, β, γ, p)
            q = flux_fnc(x, y, t, α, β, γ, p)
            u = α * x + β * y + γ
            @test all(q .≈ [-delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * α,
                -delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * β])
        end

        diffusion_function = (x, y, t, u, p) -> x * y + p
        diffusion_parameters = 3.7
        delay_function = (x, y, t, u, p) -> 1 / (1 + exp(-t))
        delay_parameters = nothing
        flux_fnc = FVM.construct_flux_function(iip_flux, flux_function, delay_function, delay_parameters, diffusion_function, diffusion_parameters)
        x, y, t, α, β, γ, p = rand(), rand(), rand(), rand(), rand(), rand(), nothing
        if iip_flux
            q = zeros(2)
            @inferred flux_fnc(q, x, y, t, α, β, γ, p)
            SHOW_WARNTYPE && @code_warntype flux_fnc(q, x, y, t, α, β, γ, p)
            u = α * x + β * y + γ
            @test all(q .≈ [-delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * α,
                -delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * β])
        else
            @inferred flux_fnc(x, y, t, α, β, γ, p)
            SHOW_WARNTYPE && @code_warntype flux_fnc(x, y, t, α, β, γ, p)
            q = flux_fnc(x, y, t, α, β, γ, p)
            u = α * x + β * y + γ
            @test all(q .≈ [-delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * α,
                -delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * β])
        end

        diffusion_function = (x, y, t, u, p) -> x * y
        diffusion_parameters = nothing
        delay_function = (x, y, t, u, p) -> 1 / (1 + exp(-t)) + p[1] * p[3]
        delay_parameters = (1.0, 0.5, 2.0)
        x, y, t, α, β, γ, p = rand(), rand(), rand(), rand(), rand(), rand(), nothing
        flux_fnc = FVM.construct_flux_function(iip_flux, flux_function, delay_function, delay_parameters, diffusion_function, diffusion_parameters)
        if iip_flux
            q = zeros(2)
            @inferred flux_fnc(q, x, y, t, α, β, γ, p)
            SHOW_WARNTYPE && @code_warntype flux_fnc(q, x, y, t, α, β, γ, p)
            u = α * x + β * y + γ
            @test all(q .≈ [-delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * α,
                -delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * β])
        else
            @inferred flux_fnc(x, y, t, α, β, γ, p)
            SHOW_WARNTYPE && @code_warntype flux_fnc(x, y, t, α, β, γ, p)
            q = flux_fnc(x, y, t, α, β, γ, p)
            u = α * x + β * y + γ
            @test all(q .≈ [-delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * α,
                -delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * β])
        end

        diffusion_function = (x, y, t, u, p) -> x * y + p
        diffusion_parameters = 3.7
        delay_function = (x, y, t, u, p) -> 1 / (1 + exp(-t)) + p[1]
        delay_parameters = 1.5
        x, y, t, α, β, γ, p = rand(), rand(), rand(), rand(), rand(), rand(), nothing
        flux_fnc = FVM.construct_flux_function(iip_flux, flux_function, delay_function, delay_parameters, diffusion_function, diffusion_parameters)
        if iip_flux
            q = zeros(2)
            @inferred flux_fnc(q, x, y, t, α, β, γ, p)
            SHOW_WARNTYPE && @code_warntype flux_fnc(q, x, y, t, α, β, γ, p)
            u = α * x + β * y + γ
            @test all(q .≈ [-delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * α,
                -delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * β])
        else
            @inferred flux_fnc(x, y, t, α, β, γ, p)
            SHOW_WARNTYPE && @code_warntype flux_fnc(x, y, t, α, β, γ, p)
            q = flux_fnc(x, y, t, α, β, γ, p)
            u = α * x + β * y + γ
            @test all(q .≈ [-delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * α,
                -delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * β])
        end

        flux_function = (x, y, t, α, β, γ, p) -> x * y * t + α * β * γ + p[1]
        flux_fnc = FVM.construct_flux_function(iip_flux, flux_function, delay_function, delay_parameters, diffusion_function, diffusion_parameters)
        @test flux_fnc === flux_function
end

## Make sure the reaction function is being constructed correctly
reaction_function = (x, y, t, u, p) -> u * (1 - u / p)
reaction_parameters = 1.2
delay_function = (x, y, t, u, p) -> x * y * t
delay_parameters = 5.0
reaction_fnc = FVM.construct_reaction_function(reaction_function, reaction_parameters,
    delay_function, delay_parameters)
x, y, t, u, p = rand(), rand(), rand(), rand(), nothing
@inferred reaction_fnc(x, y, t, u, nothing)
SHOW_WARNTYPE && @code_warntype reaction_fnc(x, y, t, u, nothing)
@test reaction_fnc(x, y, t, u, nothing) ≈ reaction_function(x, y, t, u, reaction_parameters) * delay_function(x, y, t, u, delay_parameters)

delay_function = nothing
delay_parameters = nothing
reaction_fnc = FVM.construct_reaction_function(reaction_function, reaction_parameters,
    delay_function, delay_parameters)
@test reaction_fnc === reaction_function

reaction_function = nothing
reaction_parameters = 1.2
delay_function = (x, y, t, u, p) -> x * y * t
delay_parameters = 5.0
reaction_fnc = FVM.construct_reaction_function(reaction_function, reaction_parameters,
    delay_function, delay_parameters)
x, y, t, u, p = rand(), rand(), rand(), rand(), nothing
@inferred reaction_fnc(x, y, t, u, nothing)
SHOW_WARNTYPE && @code_warntype reaction_fnc(x, y, t, u, nothing)
@test reaction_fnc(x, y, t, u, nothing) ≈ 0.0
@test reaction_fnc(x, y, t, Float32(0), nothing) === 0.0f0

reaction_function = nothing
reaction_parameters = nothing
delay_function = nothing
delay_parameters = nothing
reaction_fnc = FVM.construct_reaction_function(reaction_function, reaction_parameters,
    delay_function, delay_parameters)
x, y, t, u, p = rand(), rand(), rand(), rand(), nothing
@inferred reaction_fnc(x, y, t, u, nothing)
SHOW_WARNTYPE && @code_warntype reaction_fnc(x, y, t, u, nothing)
@test reaction_fnc(x, y, t, u, nothing) ≈ 0.0
@test reaction_fnc(x, y, t, Float32(0), nothing) === 0.0f0

## Now check that the object is constructed correctly 
a, b, c, d, nx, ny, T, adj, adj2v, DG, pts, BN = example_triangulation()
for coordinate_type in (NTuple{2,Float64}, SVector{2,Float64})
    for control_volume_storage_type_vector in (Vector{coordinate_type}, SVector{3,coordinate_type})
        for control_volume_storage_type_scalar in (Vector{Float64}, SVector{3,Float64})
            for shape_function_coefficient_storage_type in (Vector{Float64}, NTuple{9,Float64})
                for interior_edge_storage_type in (Vector{Int64}, NTuple{2,Int64},)
                    for interior_edge_pair_storage_type in (Vector{interior_edge_storage_type}, NTuple{2,interior_edge_storage_type})
                        geo = FVMGeometry(T, adj, adj2v, DG, pts, BN;
                            coordinate_type, control_volume_storage_type_vector,
                            control_volume_storage_type_scalar, shape_function_coefficient_storage_type,
                            interior_edge_storage_type, interior_edge_pair_storage_type)
                        dirichlet_f1 = (x, y, t, u, p) -> x^2 + y^2
                        dirichlet_f2 = (x, y, t, u, p) -> x * p[1] * p[2] + u
                        neumann_f1 = (x, y, t, u, p) -> 0.0
                        neumann_f2 = (x, y, t, u, p) -> u * p[1]
                        dudt_dirichlet_f1 = (x, y, t, u, p) -> x * y + 1
                        dudt_dirichlet_f2 = (x, y, t, u, p) -> u + y * t
                        functions = (dirichlet_f2, neumann_f1, dudt_dirichlet_f1, dudt_dirichlet_f2)
                        types = (:D, :N, :dudt, :dudt)
                        boundary_node_vector = BN
                        params = ((1.0, 2.0), nothing, nothing, nothing)
                        BCs = FVM.BoundaryConditions(geo, functions, types, boundary_node_vector; params)
                        iip_flux = true
                        flux_function = (q, x, y, t, α, β, γ, p) -> (q[1] = x * y * t; q[2] = t; nothing)
                        initial_condition = zeros(20)
                        final_time = 5.0
                        prob = FVMProblem(geo, BCs; iip_flux, flux_function, initial_condition, final_time)
                        @test isinplace(prob) == iip_flux
                        @test prob.boundary_conditions == BCs
                        @test prob.flux_function == flux_function
                        @test prob.initial_condition == initial_condition
                        @test prob.mesh == geo == FVM.get_mesh(prob)
                        @test prob.reaction_parameters === nothing
                        @test prob.final_time == final_time
                        @test prob.flux_parameters === nothing
                        @test prob.initial_time == 0.0
                        @test prob.reaction_function(rand(), rand(), rand(), rand(), nothing) == 0.0
                        @inferred prob.reaction_function(rand(), rand(), rand(), rand(), nothing)
                        SHOW_WARNTYPE && @code_warntype prob.reaction_function(rand(), rand(), rand(), rand(), nothing)
                        @test prob.reaction_function(rand(), rand(), rand(), rand(Float32), nothing) == 0.0f0
                        @test prob.steady == false

                        flux_function = (q, x, y, t, α, β, γ, p) -> (q[1] = x * y * t; q[2] = p; nothing)
                        flux_parameters = 3.81
                        prob = FVMProblem(geo, BCs; iip_flux, flux_function, initial_condition, final_time, flux_parameters)
                        @test prob.flux_function == flux_function
                        @test prob.flux_parameters == flux_parameters
                        q = zeros(2)
                        x, y, t, α, β, γ = rand(6)
                        prob.flux_function(q, x, y, t, α, β, γ, flux_parameters)
                        @test q ≈ [x * y * t, flux_parameters]
                        @test prob.boundary_conditions == BCs == FVM.get_boundary_conditions(prob)
                        @test prob.flux_function == flux_function
                        @test prob.initial_condition == initial_condition
                        @test prob.mesh == geo
                        @test prob.reaction_parameters === nothing
                        @test prob.final_time == final_time
                        @test prob.flux_parameters === flux_parameters
                        @test prob.initial_time == 0.0

                        diffusion_function = (x, y, t, u, p) -> u^2 + p[1] * p[2]
                        diffusion_parameters = (2.0, 3.0)
                        reaction_function = (x, y, t, u, p) -> u * (1 - u / p[1])
                        reaction_parameters = [5.0]
                        prob = FVMProblem(geo, BCs; iip_flux, diffusion_function, diffusion_parameters, initial_condition, reaction_function, reaction_parameters, final_time, initial_time=3.71)
                        @test isinplace(prob) == iip_flux
                        @test prob.boundary_conditions == BCs == FVM.get_boundary_conditions(prob)
                        q1 = zeros(2)
                        x, y, t, α, β, γ = rand(6)
                        prob.flux_function(q1, x, y, t, α, β, γ, nothing)
                        u = α * x + β * y + γ
                        q2 = [-diffusion_function(x, y, t, u, diffusion_parameters) * α, -diffusion_function(x, y, t, u, diffusion_parameters) * β]
                        @test q1 ≈ q2
                        @test prob.reaction_function(x, y, t, u, prob.reaction_parameters) == reaction_function(x, y, t, u, reaction_parameters)
                        @inferred prob.reaction_function(x, y, t, u, prob.reaction_parameters)
                        SHOW_WARNTYPE && @code_warntype prob.reaction_function(x, y, t, u, prob.reaction_parameters)
                        @test prob.boundary_conditions == BCs
                        @test prob.initial_condition == initial_condition
                        @test prob.mesh == geo
                        @test prob.reaction_parameters === reaction_parameters
                        @test prob.final_time == final_time
                        @test prob.flux_parameters === nothing
                        @test prob.initial_time == 3.71

                        delay_function = (x, y, t, u, p) -> 1 / (1 + exp(-t * p[1]))
                        delay_parameters = [2.371]
                        prob = FVMProblem(geo, BCs; iip_flux, diffusion_function, diffusion_parameters, initial_condition, delay_function, delay_parameters, reaction_function, reaction_parameters, final_time, initial_time=3.71)
                        @test isinplace(prob) == iip_flux
                        @test prob.boundary_conditions == BCs
                        q1 = zeros(2)
                        x, y, t, α, β, γ = rand(6)
                        prob.flux_function(q1, x, y, t, α, β, γ, nothing)
                        SHOW_WARNTYPE && @code_warntype prob.flux_function(q1, x, y, t, α, β, γ, nothing)
                        u = α * x + β * y + γ
                        q2 = [-delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * α, -delay_function(x, y, t, u, delay_parameters) * diffusion_function(x, y, t, u, diffusion_parameters) * β]
                        @test q1 ≈ q2
                        @test prob.reaction_function(x, y, t, u, prob.reaction_parameters) ≈ delay_function(x, y, t, u, delay_parameters) * reaction_function(x, y, t, u, reaction_parameters)
                        @test prob.initial_condition == initial_condition
                        @test prob.mesh == geo
                        @test prob.reaction_parameters === reaction_parameters
                        @test prob.final_time == final_time
                        @test prob.flux_parameters === nothing
                        @test prob.initial_time == 3.71
                        @test prob.steady == false

                        ## Test some of the getters 
                        for V in T
                            @test FVM.gets(prob, V) == prob.mesh.element_information_list[V].shape_function_coefficients
                            for i in 1:9
                                @test FVM.gets(prob, V, i) == prob.mesh.element_information_list[V].shape_function_coefficients[i]
                            end
                            @test FVM.get_midpoints(prob, V) == prob.mesh.element_information_list[V].midpoints
                            for i in 1:3
                                @test FVM.get_midpoints(prob, V, i) == prob.mesh.element_information_list[V].midpoints[i]
                            end
                            @test FVM.get_control_volume_edge_midpoints(prob, V) == prob.mesh.element_information_list[V].control_volume_edge_midpoints
                            for i in 1:3
                                @test FVM.get_control_volume_edge_midpoints(prob, V, i) == prob.mesh.element_information_list[V].control_volume_edge_midpoints[i]
                            end
                            @test FVM.get_normals(prob, V) == prob.mesh.element_information_list[V].normals
                            for i in 1:3
                                @test FVM.get_normals(prob, V, i) == prob.mesh.element_information_list[V].normals[i]
                            end
                            @test FVM.get_lengths(prob, V) == prob.mesh.element_information_list[V].lengths
                            for i in 1:3
                                @test FVM.get_lengths(prob, V, i) == prob.mesh.element_information_list[V].lengths[i]
                            end
                        end
                        x, y, t, α, β, γ = rand(6)
                        flux_cache = zeros(2)
                        FVM.get_flux!(flux_cache, prob, x, y, t, α, β, γ)
                        flux_cache_2 = zeros(2)
                        prob.flux_function(flux_cache_2, x, y, t, α, β, γ, prob.flux_parameters)
                        @test all(flux_cache .≈ flux_cache_2)
                        prob = FVMProblem(geo, BCs; iip_flux=false, diffusion_function, diffusion_parameters, initial_condition, delay_function, delay_parameters, reaction_function, reaction_parameters, final_time, initial_time=3.71)
                        flux_cache = FVM.get_flux(prob, x, y, t, α, β, γ)
                        @test all(flux_cache .≈ flux_cache_2)
                        SHOW_WARNTYPE && @code_warntype FVM.get_flux(prob, x, y, t, α, β, γ)
                        @test FVM.get_interior_or_neumann_nodes(prob) == prob.boundary_conditions.interior_or_neumann_nodes
                        for j ∈ FVM.get_interior_or_neumann_nodes(prob)
                            @test FVM.is_interior_or_neumann_node(prob, j)
                        end
                        j = rand(Int64, 500)
                        setdiff!(j, FVM.get_interior_or_neumann_nodes(prob))
                        for j ∈ j
                            @test !FVM.is_interior_or_neumann_node(prob, j)
                        end
                        @test FVM.get_interior_elements(prob) == prob.mesh.interior_information.elements
                        true_interior_edge_boundary_element_identifier = get_interior_identifier_for_example_triangulation(interior_edge_pair_storage_type)
                        for (V, E) in true_interior_edge_boundary_element_identifier
                            @test FVM.get_interior_edges(prob, V) == E
                        end
                        @test FVM.get_boundary_elements(prob) == prob.mesh.boundary_information.boundary_elements
                        for j in axes(pts, 2)
                            @test get_point(prob, j) == Tuple(pts[:, j])
                            @test FVM.get_volumes(prob, j) == prob.mesh.volumes[j]
                        end
                        x, y, t, u = rand(4)
                        @test FVM.get_reaction(prob, x, y, t, u) == prob.reaction_function(x, y, t, u, prob.reaction_parameters)
                        @inferred FVM.get_reaction(prob, x, y, t, u)
                        SHOW_WARNTYPE && @code_warntype FVM.get_reaction(prob, x, y, t, u)
                        @test FVM.get_dirichlet_nodes(prob) == prob.boundary_conditions.dirichlet_nodes
                        @test FVM.get_neumann_nodes(prob) == prob.boundary_conditions.neumann_nodes
                        @test FVM.get_dudt_nodes(prob) == prob.boundary_conditions.dudt_nodes
                        @test FVM.get_boundary_nodes(prob) == prob.mesh.boundary_information.boundary_nodes
                        for i in eachindex(prob.boundary_conditions.parameters)
                            @test FVM.get_boundary_function_parameters(prob, i) == prob.boundary_conditions.parameters[i]
                        end
                        for j in FVM.get_boundary_nodes(prob)
                            @test FVM.map_node_to_segment(prob, j) == prob.boundary_conditions.type_map[j]
                        end
                        x, y, t, u = rand(4)
                        @test FVM.evaluate_boundary_function(prob, 1, x, y, t, u) ≈ prob.boundary_conditions.functions[1](x, y, t, u, FVM.get_boundary_function_parameters(prob, 1))
                        @test FVM.evaluate_boundary_function(prob, 2, x, y, t, u) ≈ prob.boundary_conditions.functions[2](x, y, t, u, FVM.get_boundary_function_parameters(prob, 2))
                        @test FVM.evaluate_boundary_function(prob, 3, x, y, t, u) ≈ prob.boundary_conditions.functions[3](x, y, t, u, FVM.get_boundary_function_parameters(prob, 3))
                        @test FVM.evaluate_boundary_function(prob, 4, x, y, t, u) ≈ prob.boundary_conditions.functions[4](x, y, t, u, FVM.get_boundary_function_parameters(prob, 4))
                        @test FVM.get_neighbours(prob) == DG
                        @test FVM.get_initial_condition(prob) == prob.initial_condition
                        @test FVM.get_initial_time(prob) == prob.initial_time
                        @test FVM.get_final_time(prob) == prob.final_time
                        @test FVM.get_time_span(prob) == (prob.initial_time, prob.final_time)
                        @test FVM.get_points(prob) == pts
                        @test num_points(prob) == size(pts, 2)
                        @test FVM.num_boundary_edges(prob) == length(FVM.get_boundary_nodes(prob))
                        @test FVM.get_adjacent(prob) == adj
                        @test FVM.get_adjacent2vertex(prob) == adj2v
                        @test FVM.get_elements(prob) == T
                        @test FVM.get_element_type(prob) == NTuple{3,Int64}
                    end
                end
            end
        end
    end
end