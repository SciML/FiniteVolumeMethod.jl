# Allocation tests for FiniteVolumeMethod.jl
# These tests verify that the core hotpath functions remain allocation-free
# to prevent performance regressions.

using FiniteVolumeMethod
using DelaunayTriangulation
using Test

@testset "Allocation Tests - FVMProblem" begin
    # Set up a test problem
    n = 20
    α = π / 4
    x₁ = [0.0, 1.0]
    y₁ = [0.0, 0.0]
    r₂ = fill(1, n)
    θ₂ = LinRange(0, α, n)
    x₂ = @. r₂ * cos(θ₂)
    y₂ = @. r₂ * sin(θ₂)
    x₃ = [cos(α), 0.0]
    y₃ = [sin(α), 0.0]
    x = [x₁, x₂, x₃]
    y = [y₁, y₂, y₃]
    boundary_nodes, points = convert_boundary_points_to_indices(x, y)
    tri = triangulate(points; boundary_nodes)
    refine!(tri; max_area = 0.001)

    mesh = FVMGeometry(tri)
    lower_bc = arc_bc = upper_bc = (x, y, t, u, p) -> zero(u)
    types = (Neumann, Dirichlet, Neumann)
    BCs = BoundaryConditions(mesh, (lower_bc, arc_bc, upper_bc), types)
    f = (x, y) -> 1 - sqrt(x^2 + y^2)
    D = (x, y, t, u, p) -> one(u)
    initial_condition = [f(x, y) for (x, y) in DelaunayTriangulation.each_point(tri)]
    final_time = 0.1
    prob = FVMProblem(mesh, BCs; diffusion_function = D, initial_condition, final_time)

    # Get parameters for testing
    params_serial = FiniteVolumeMethod.get_serial_parameters(prob)

    # Set up du and u
    u = copy(initial_condition)
    du = similar(u)
    t = 0.0

    # Warmup
    FiniteVolumeMethod.serial_fvm_eqs!(du, u, prob, t)
    FiniteVolumeMethod.fvm_eqs!(du, u, params_serial, t)

    @testset "serial_fvm_eqs! is allocation-free" begin
        fill!(du, 0.0)
        num_allocs = @allocated FiniteVolumeMethod.serial_fvm_eqs!(du, u, prob, t)
        @test num_allocs == 0
    end

    @testset "fvm_eqs! (serial) is allocation-free" begin
        fill!(du, 0.0)
        num_allocs = @allocated FiniteVolumeMethod.fvm_eqs!(du, u, params_serial, t)
        @test num_allocs == 0
    end

    @testset "get_triangle_contributions! is allocation-free" begin
        fill!(du, 0.0)
        FiniteVolumeMethod.get_triangle_contributions!(du, u, prob, t)
        fill!(du, 0.0)
        num_allocs = @allocated FiniteVolumeMethod.get_triangle_contributions!(du, u, prob, t)
        @test num_allocs == 0
    end

    @testset "get_boundary_edge_contributions! is allocation-free" begin
        fill!(du, 0.0)
        FiniteVolumeMethod.get_boundary_edge_contributions!(du, u, prob, t)
        fill!(du, 0.0)
        num_allocs = @allocated FiniteVolumeMethod.get_boundary_edge_contributions!(du, u, prob, t)
        @test num_allocs == 0
    end

    @testset "get_source_contributions! is allocation-free" begin
        fill!(du, 0.0)
        FiniteVolumeMethod.get_source_contributions!(du, u, prob, t)
        fill!(du, 0.0)
        num_allocs = @allocated FiniteVolumeMethod.get_source_contributions!(du, u, prob, t)
        @test num_allocs == 0
    end
end

@testset "Allocation Tests - FVMSystem" begin
    # Set up a test FVMSystem problem
    tri = triangulate_rectangle(0, 100, 0, 100, 10, 10, single_boundary = true)

    mesh = FVMGeometry(tri)
    bc_u = (x, y, t, (u_val, v_val), p) -> zero(u_val)
    bc_v = (x, y, t, (u_val, v_val), p) -> zero(v_val)
    BCs_u = BoundaryConditions(mesh, bc_u, Neumann)
    BCs_v = BoundaryConditions(mesh, bc_v, Neumann)
    q_u = (x, y, t, (αu, αv), (βu, βv), (γu, γv), p) -> begin
        u_local = αu * x + βu * y + γu
        ∇u = (αu, βu)
        ∇v = (αv, βv)
        χu = p.c * u_local / (1 + u_local^2)
        _q = χu .* ∇v .- ∇u
        return _q
    end
    q_v = (x, y, t, (αu, αv), (βu, βv), (γu, γv), p) -> begin
        ∇v = (αv, βv)
        _q = -p.D .* ∇v
        return _q
    end
    S_u = (x, y, t, (u_val, v_val), p) -> begin
        return u_val * (1 - u_val)
    end
    S_v = (x, y, t, (u_val, v_val), p) -> begin
        return u_val - p.a * v_val
    end
    q_u_parameters = (c = 4.0,)
    q_v_parameters = (D = 1.0,)
    S_v_parameters = (a = 0.1,)
    u_initial_condition = 0.01rand(DelaunayTriangulation.num_solid_vertices(tri))
    v_initial_condition = zeros(DelaunayTriangulation.num_solid_vertices(tri))
    final_time = 1000.0
    u_prob = FVMProblem(
        mesh, BCs_u;
        flux_function = q_u, flux_parameters = q_u_parameters,
        source_function = S_u,
        initial_condition = u_initial_condition, final_time = final_time
    )
    v_prob = FVMProblem(
        mesh, BCs_v;
        flux_function = q_v, flux_parameters = q_v_parameters,
        source_function = S_v, source_parameters = S_v_parameters,
        initial_condition = v_initial_condition, final_time = final_time
    )
    sys_prob = FVMSystem(u_prob, v_prob)

    # Get parameters
    params_serial = FiniteVolumeMethod.get_serial_parameters(sys_prob)

    # Set up du and u_sys
    u_sys = copy(sys_prob.initial_condition)
    du_sys = similar(u_sys)
    t_sys = 0.0

    # Warmup
    FiniteVolumeMethod.serial_fvm_eqs!(du_sys, u_sys, sys_prob, t_sys)
    FiniteVolumeMethod.fvm_eqs!(du_sys, u_sys, params_serial, t_sys)

    @testset "serial_fvm_eqs! (FVMSystem) is allocation-free" begin
        fill!(du_sys, 0.0)
        num_allocs = @allocated FiniteVolumeMethod.serial_fvm_eqs!(du_sys, u_sys, sys_prob, t_sys)
        @test num_allocs == 0
    end

    @testset "fvm_eqs! (FVMSystem serial) is allocation-free" begin
        fill!(du_sys, 0.0)
        num_allocs = @allocated FiniteVolumeMethod.fvm_eqs!(du_sys, u_sys, params_serial, t_sys)
        @test num_allocs == 0
    end
end
