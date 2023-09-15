using ..FiniteVolumeMethod
using Test
using LinearAlgebra
using DelaunayTriangulation
using StructEquality
using OrdinaryDiffEq
using LinearSolve
const FVM = FiniteVolumeMethod
const DT = DelaunayTriangulation
@struct_hash_equal FVM.Conditions
include("test_functions.jl")

@testset "Some generic tests" begin
    prob, tri, mesh, BCs, ICs,
    flux_function, flux_parameters,
    source_function, source_parameters,
    initial_condition = example_problem()
    @test sprint(show, MIME"text/plain"(), prob) == "FVMProblem with $(num_points(tri)) nodes and time span ($(prob.initial_time), $(prob.final_time))"
    conds = FVM.Conditions(mesh, BCs, ICs)
    @test prob.mesh == mesh
    @test prob.conditions == conds
    @test prob.flux_function == flux_function
    @test prob.flux_parameters == flux_parameters
    @test prob.source_function == source_function
    @test prob.source_parameters == source_parameters
    @test prob.initial_condition == initial_condition
    @test prob.initial_time == 2.0
    @test prob.final_time == 5.0
    x, y, t, α, β, γ, p = 0.5, -1.0, 2.3, 0.371, -5.37, 17.5, flux_parameters
    u = α * x + β * y + γ
    qx = -α * u * p[1] + t
    qy = x + t - β * u * p[2]
    @test FVM.eval_flux_function(prob, x, y, t, α, β, γ) == (qx, qy)
    @inferred FVM.eval_flux_function(prob, x, y, t, α, β, γ)
    @test FVM._neqs(prob) == 0
    @test !FVM.is_system(prob)
    @test FVM.has_dirichlet_nodes(prob)
    for (i, idx) in prob.conditions.dirichlet_nodes
        @test FVM.get_dirichlet_fidx(prob, i) == idx
    end
    @test FVM.get_dirichlet_nodes(prob) == prob.conditions.dirichlet_nodes
    for (e, idx) in prob.conditions.neumann_edges
        @test FVM.get_neumann_fidx(prob, e...) == idx
    end
    for e in each_edge(prob.mesh.triangulation)
        if e ∉ keys(prob.conditions.neumann_edges)
            @test !FVM.is_neumann_edge(prob, e...)
        else
            @test FVM.is_neumann_edge(prob, e...)
        end
    end
    for i in each_point_index(prob.mesh.triangulation)
        if i ∈ keys(prob.conditions.dirichlet_nodes)
            @test FVM.has_condition(prob, i)
            @test FVM.is_dirichlet_node(prob, i)
        else
            @test !FVM.has_condition(prob, i)
            @test !FVM.is_dirichlet_node(prob, i)
        end
    end
    x, y, t, u = 0.2, 0.3, 0.4, 0.5
    @test FVM.eval_condition_fnc(prob, 1, x, y, t, u) ≈ x + y + t + u + 0.29
    @test FVM.eval_condition_fnc(prob, 2, x, y, t, u) ≈ x * y + u - 0.5
    @test FVM.eval_condition_fnc(prob, 3, x, y, t, u) ≈ u + 0.2 - t
    @test FVM.eval_condition_fnc(prob, 4, x, y, t, u) ≈ x
    @test FVM.eval_condition_fnc(prob, 5, x, y, t, u) ≈ y - x
    @test FVM.eval_source_fnc(prob, x, y, t, u) ≈ u + 1.5
    i, j, k = first(each_solid_triangle(prob.mesh.triangulation))
    @test FVM.get_triangle_props(prob, i, j, k) == prob.mesh.triangle_props[(i, j, k)]
    @test get_point(prob, i) == get_point(prob.mesh.triangulation, i)
    @test FVM.get_volume(prob, i) == prob.mesh.cv_volumes[i]

    x, y, t, α, β, γ, p = 0.5, -1.0, 2.3, 0.371, -5.37, 17.5, flux_parameters
    u = α * x + β * y + γ
    qx = -α * u * p[1] + t
    qy = x + t - β * u * p[2]
    steady = SteadyFVMProblem(prob)
    @test sprint(show, MIME"text/plain"(), steady) == "SteadyFVMProblem with $(num_points(tri)) nodes"
    @inferred SteadyFVMProblem(prob)
    @test FVM.eval_flux_function(steady, x, y, t, α, β, γ) == (qx, qy)
    @inferred FVM.eval_flux_function(steady, x, y, t, α, β, γ)
    @test FVM._neqs(steady) == 0
    @test !FVM.is_system(steady)

    prob1, prob2, prob3, prob4, prob5 = example_problem(1; tri, mesh, initial_condition)[1],
    example_problem(2; tri, mesh, initial_condition)[1],
    example_problem(3; tri, mesh, initial_condition)[1],
    example_problem(4; tri, mesh, initial_condition)[1],
    example_problem(5; tri, mesh, initial_condition)[1]
    system = FVMSystem(prob1, prob2, prob3, prob4, prob5)
    @test sprint(show, MIME"text/plain"(), system) == "FVMSystem with 5 equations and time span ($(system.initial_time), $(system.final_time))"
    @inferred FVMSystem(prob1, prob2, prob3, prob4, prob5)
    _α = ntuple(_ -> α, 5)
    _β = ntuple(_ -> β, 5)
    _γ = ntuple(_ -> γ, 5)
    @test FVM.eval_flux_function(system, x, y, t, _α, _β, _γ) == ntuple(_ -> (qx, qy), 5)
    @inferred FVM.eval_flux_function(system, x, y, t, _α, _β, _γ)
    @test system.initial_condition ≈ [initial_condition initial_condition initial_condition initial_condition initial_condition]'
    @test FVM._neqs(system) == 5
    @test FVM.is_system(system)
    @test system.initial_time == 2.0
    @test system.final_time == 5.0
    @test FVM.get_conditions(system, 1) == system.conditions[1]
    @test FVM.get_conditions(system, 2) == system.conditions[2]
    @test FVM.get_conditions(system, 3) == system.conditions[3]
    @test FVM.get_conditions(system, 4) == system.conditions[4]
    @test FVM.get_conditions(system, 5) == system.conditions[5]
    @test system.conditions[1].neumann_edges == prob1.conditions.neumann_edges
    @test system.conditions[2].neumann_edges == prob2.conditions.neumann_edges
    @test system.conditions[3].neumann_edges == prob3.conditions.neumann_edges
    @test system.conditions[4].neumann_edges == prob4.conditions.neumann_edges
    @test system.conditions[5].neumann_edges == prob5.conditions.neumann_edges
    @test system.conditions[1].dirichlet_nodes == prob1.conditions.dirichlet_nodes
    @test system.conditions[2].dirichlet_nodes == prob2.conditions.dirichlet_nodes
    @test system.conditions[3].dirichlet_nodes == prob3.conditions.dirichlet_nodes
    @test system.conditions[4].dirichlet_nodes == prob4.conditions.dirichlet_nodes
    @test system.conditions[5].dirichlet_nodes == prob5.conditions.dirichlet_nodes
    @test system.conditions[1].constrained_edges == prob1.conditions.constrained_edges
    @test system.conditions[2].constrained_edges == prob2.conditions.constrained_edges
    @test system.conditions[3].constrained_edges == prob3.conditions.constrained_edges
    @test system.conditions[4].constrained_edges == prob4.conditions.constrained_edges
    @test system.conditions[5].constrained_edges == prob5.conditions.constrained_edges
    @test system.conditions[1].dudt_nodes == prob1.conditions.dudt_nodes
    @test system.conditions[2].dudt_nodes == prob2.conditions.dudt_nodes
    @test system.conditions[3].dudt_nodes == prob3.conditions.dudt_nodes
    @test system.conditions[4].dudt_nodes == prob4.conditions.dudt_nodes
    @test system.conditions[5].dudt_nodes == prob5.conditions.dudt_nodes
    @test system.functions == (prob1.conditions.functions...,
        prob2.conditions.functions...,
        prob3.conditions.functions...,
        prob4.conditions.functions...,
        prob5.conditions.functions...)

    steady_system = SteadyFVMProblem(system)
    @inferred SteadyFVMProblem(system)
    @test FVM.eval_flux_function(steady_system, x, y, t, _α, _β, _γ) == ntuple(_ -> (qx, qy), 5)
    @inferred FVM.eval_flux_function(steady_system, x, y, t, _α, _β, _γ)
    @test FVM._neqs(steady_system) == 5
    @test FVM.is_system(steady_system)

    _q = FVM.construct_flux_function(flux_function, nothing, nothing)
    @test _q == flux_function
    D = (x, y, t, u, p) -> x + y + t + u + p[2]
    Dp = (0.2, 5.73)
    _q = FVM.construct_flux_function(nothing, D, Dp)
    x, y, t, α, β, γ = rand(6)
    u = α * x + β * y + γ
    @test _q(x, y, t, α, β, γ, nothing) == (-α, -β) .* D(x, y, t, u, Dp)
    @inferred _q(x, y, t, α, β, γ, nothing)

    test_compute_flux(prob, steady, system, steady_system)

    for prob in (prob, steady, system, steady_system)
        test_jacobian_sparsity(prob)
        @inferred FVM.jacobian_sparsity(prob)
    end
end

@testset "FVMSystem" begin
    tri = triangulate_rectangle(0, 1, 0, 1, 10, 10, single_boundary=false)
    mesh = FVMGeometry(tri)
    Φ_bot = (x, y, t, u, p) -> -1 / 4 * exp(-x - t / 2)
    Φ_right = (x, y, t, u, p) -> 1 / 4 * exp(-1 - y - t / 2)
    Φ_top = (x, y, t, u, p) -> exp(-1 - x - t / 2)
    Φ_left = (x, y, t, u, p) -> -1 / 4 * exp(-y - t / 2)
    Φ_bc_fncs = (Φ_bot, Φ_right, Φ_top, Φ_left)
    Φ_bc_types = (Neumann, Neumann, Dirichlet, Neumann)
    Φ_BCs = BoundaryConditions(mesh, Φ_bc_fncs, Φ_bc_types)
    Ψ_bot = (x, y, t, u, p) -> exp(x + t / 2)
    Ψ_right = (x, y, t, u, p) -> -1 / 4 * exp(1 + y + t / 2)
    Ψ_top = (x, y, t, u, p) -> -1 / 4 * exp(1 + x + t / 2)
    Ψ_left = (x, y, t, u, p) -> exp(y + t / 2)
    Ψ_bc_fncs = (Ψ_bot, Ψ_right, Ψ_top, Ψ_left)
    Ψ_bc_types = (Dirichlet, Neumann, Neumann, Dirichlet)
    Ψ_BCs = BoundaryConditions(mesh, Ψ_bc_fncs, Ψ_bc_types)
    Φ_q = (x, y, t, α, β, γ, p) -> (-α[1] / 4, -β[1] / 4)
    Ψ_q = (x, y, t, α, β, γ, p) -> (-α[2] / 4, -β[2] / 4)
    Φ_S = (x, y, t, (Φ, Ψ), p) -> Φ^2 * Ψ - 2Φ
    Ψ_S = (x, y, t, (Φ, Ψ), p) -> -Φ^2 * Ψ + Φ
    Φ_exact = (x, y, t) -> exp(-x - y - t / 2)
    Ψ_exact = (x, y, t) -> exp(x + y + t / 2)
    Φ₀ = [Φ_exact(x, y, 0) for (x, y) in each_point(tri)]
    Ψ₀ = [Ψ_exact(x, y, 0) for (x, y) in each_point(tri)]
    Φ_prob = FVMProblem(mesh, Φ_BCs; flux_function=Φ_q, source_function=Φ_S,
        initial_condition=Φ₀, final_time=5.0)
    Ψ_prob = FVMProblem(mesh, Ψ_BCs; flux_function=Ψ_q, source_function=Ψ_S,
        initial_condition=Ψ₀, final_time=5.0)
    prob = FVMSystem(Φ_prob, Ψ_prob)

    @test prob.mesh === mesh
    @test prob.initial_condition == [Φ₀'; Ψ₀']
    @test prob.initial_time == 0.0
    @test prob.final_time == 5.0
    cond1 = FVM.SimpleConditions(Φ_prob.conditions.neumann_edges, Φ_prob.conditions.constrained_edges, Φ_prob.conditions.dirichlet_nodes, Φ_prob.conditions.dudt_nodes)
    cond2 = FVM.SimpleConditions(Ψ_prob.conditions.neumann_edges, Ψ_prob.conditions.constrained_edges, Ψ_prob.conditions.dirichlet_nodes, Ψ_prob.conditions.dudt_nodes)
    syscond = (cond1, cond2)
    @test prob.conditions == syscond
    @test prob.cnum_fncs == (0, 4)
    @test prob.functions == (Φ_prob.conditions.functions..., Ψ_prob.conditions.functions...)
    source_fncs = (Φ_S, Ψ_S)
    source_params = (nothing, nothing)
    wrapped_source_fncs = FVM.wrap_functions(source_fncs, source_params)
    flux_fncs = (Φ_q, Ψ_q)
    flux_params = (nothing, nothing)
    wrapped_flux_fncs = FVM.wrap_functions(flux_fncs, flux_params)
    @test prob.source_functions == wrapped_source_fncs
    @test prob.flux_functions == wrapped_flux_fncs

    @test FVM.map_fidx(prob, 1, 1) == 1
    @test FVM.map_fidx(prob, 2, 1) == 2
    @test FVM.map_fidx(prob, 3, 1) == 3
    @test FVM.map_fidx(prob, 4, 1) == 4
    @test FVM.map_fidx(prob, 1, 2) == 5
    @test FVM.map_fidx(prob, 2, 2) == 6
    @test FVM.map_fidx(prob, 3, 2) == 7
    @test FVM.map_fidx(prob, 4, 2) == 8

    for ((i, j), fidx) in cond1.neumann_edges
        @test FVM.get_neumann_fidx(prob, i, j, 1) == fidx

        x, y, t, u = rand(4)
        @test Φ_prob.conditions.functions[fidx](x, y, t, u) == prob.functions[fidx](x, y, t, u)
        @test FVM.eval_condition_fnc(prob, fidx, 1, x, y, t, u) == prob.functions[fidx](x, y, t, u)
        @inferred FVM.eval_condition_fnc(prob, fidx, 1, x, y, t, u)

        @test FVM.is_neumann_edge(prob, i, j, 1)
    end
    for ((i, j), fidx) in cond2.neumann_edges
        @test FVM.get_neumann_fidx(prob, i, j, 2) == fidx

        x, y, t, u = rand(4)
        @test Ψ_prob.conditions.functions[fidx](x, y, t, u) == prob.functions[fidx+4](x, y, t, u)
        @test FVM.eval_condition_fnc(prob, fidx, 2, x, y, t, u) == prob.functions[fidx+4](x, y, t, u)
        @inferred FVM.eval_condition_fnc(prob, fidx, 2, x, y, t, u)

        @test FVM.is_neumann_edge(prob, i, j, 2)
    end
    for (i, fidx) in cond1.dirichlet_nodes
        @test FVM.get_dirichlet_fidx(prob, i, 1) == fidx

        x, y, t, u = rand(4)
        @test Φ_prob.conditions.functions[fidx](x, y, t, u) == prob.functions[fidx](x, y, t, u)
        @test FVM.eval_condition_fnc(prob, fidx, 1, x, y, t, u) == prob.functions[fidx](x, y, t, u)
        @inferred FVM.eval_condition_fnc(prob, fidx, 1, x, y, t, u)

        @test FVM.is_dirichlet_node(prob, i, 1)
    end
    for (i, fidx) in cond2.dirichlet_nodes
        @test FVM.get_dirichlet_fidx(prob, i, 2) == fidx

        x, y, t, u = rand(4)
        @test Ψ_prob.conditions.functions[fidx](x, y, t, u) == prob.functions[fidx+4](x, y, t, u)
        @test FVM.eval_condition_fnc(prob, fidx, 2, x, y, t, u) == prob.functions[fidx+4](x, y, t, u)
        @inferred FVM.eval_condition_fnc(prob, fidx, 2, x, y, t, u)

        @test FVM.is_dirichlet_node(prob, i, 2)
    end
    for i in each_solid_vertex(tri)
        x, y, t = rand(3)
        u = Tuple(rand(2))
        @test FVM.eval_source_fnc(prob, 1, x, y, t, u) == Φ_S(x, y, t, u, nothing)
        @test FVM.eval_source_fnc(prob, 2, x, y, t, u) == Ψ_S(x, y, t, u, nothing)
    end
    for (i, j) in each_edge(tri)
        if (i, j) ∉ keys(cond1.neumann_edges) && (i, j) ∉ keys(cond2.neumann_edges)
            @test !FVM.is_neumann_edge(prob, i, j, 1)
            @test !FVM.is_neumann_edge(prob, i, j, 2)
        elseif (i, j) ∈ keys(cond1.neumann_edges) && (i, j) ∉ keys(cond2.neumann_edges)
            @test FVM.is_neumann_edge(prob, i, j, 1)
            @test !FVM.is_neumann_edge(prob, i, j, 2)
        elseif (i, j) ∉ keys(cond1.neumann_edges) && (i, j) ∈ keys(cond2.neumann_edges)
            @test !FVM.is_neumann_edge(prob, i, j, 1)
            @test FVM.is_neumann_edge(prob, i, j, 2)
        else
            @test FVM.is_neumann_edge(prob, i, j, 1)
            @test FVM.is_neumann_edge(prob, i, j, 2)
        end
        @test !FVM.is_constrained_edge(prob, i, j, 1)
        @test !FVM.is_constrained_edge(prob, i, j, 2)
    end
    for i in each_point_index(tri)
        if i ∉ keys(cond1.dirichlet_nodes) && i ∉ keys(cond2.dirichlet_nodes)
            @test !FVM.is_dirichlet_node(prob, i, 1)
            @test !FVM.is_dirichlet_node(prob, i, 2)
            @test !FVM.has_condition(prob, i, 1)
            @test !FVM.has_condition(prob, i, 2)
        elseif i ∈ keys(cond1.dirichlet_nodes) && i ∉ keys(cond2.dirichlet_nodes)
            @test FVM.is_dirichlet_node(prob, i, 1)
            @test !FVM.is_dirichlet_node(prob, i, 2)
            @test FVM.has_condition(prob, i, 1)
            @test !FVM.has_condition(prob, i, 2)
        elseif i ∉ keys(cond1.dirichlet_nodes) && i ∈ keys(cond2.dirichlet_nodes)
            @test !FVM.is_dirichlet_node(prob, i, 1)
            @test FVM.is_dirichlet_node(prob, i, 2)
            @test !FVM.has_condition(prob, i, 1)
            @test FVM.has_condition(prob, i, 2)
        else
            @test FVM.is_dirichlet_node(prob, i, 1)
            @test FVM.is_dirichlet_node(prob, i, 2)
            @test FVM.has_condition(prob, i, 1)
            @test FVM.has_condition(prob, i, 2)
        end
        @test !FVM.is_dudt_node(prob, i, 1)
        @test !FVM.is_dudt_node(prob, i, 2)
    end
    @test FVM.has_dirichlet_nodes(prob, 1)
    @test FVM.has_dirichlet_nodes(prob, 2)
    @test FVM.has_dirichlet_nodes(prob)
    @test FVM.get_dirichlet_nodes(prob, 1) == prob.conditions[1].dirichlet_nodes
    @test FVM.get_dirichlet_nodes(prob, 2) == prob.conditions[2].dirichlet_nodes
    x, y, t = rand(3)
    α = Tuple(rand(2))
    β = Tuple(rand(2))
    γ = Tuple(rand(2))
    _q = FVM.eval_flux_function(prob, x, y, t, α, β, γ)
    @inferred FVM.eval_flux_function(prob, x, y, t, α, β, γ)
    @test _q == (Φ_q(x, y, t, α, β, γ, nothing), Ψ_q(x, y, t, α, β, γ, nothing))
end


tri = triangulate_rectangle(0, 1, 0, 1, 100, 100, single_boundary=false)
mesh = FVMGeometry(tri)
Φ_bot = (x, y, t, u, p) -> -1 / 4 * exp(-x - t / 2)
Φ_right = (x, y, t, u, p) -> 1 / 4 * exp(-1 - y - t / 2)
Φ_top = (x, y, t, u, p) -> exp(-1 - x - t / 2)
Φ_left = (x, y, t, u, p) -> -1 / 4 * exp(-y - t / 2)
Φ_bc_fncs = (Φ_bot, Φ_right, Φ_top, Φ_left)
Φ_bc_types = (Neumann, Neumann, Dirichlet, Neumann)
Φ_BCs = BoundaryConditions(mesh, Φ_bc_fncs, Φ_bc_types)
Ψ_bot = (x, y, t, u, p) -> exp(x + t / 2)
Ψ_right = (x, y, t, u, p) -> -1 / 4 * exp(1 + y + t / 2)
Ψ_top = (x, y, t, u, p) -> -1 / 4 * exp(1 + x + t / 2)
Ψ_left = (x, y, t, u, p) -> exp(y + t / 2)
Ψ_bc_fncs = (Ψ_bot, Ψ_right, Ψ_top, Ψ_left)
Ψ_bc_types = (Dirichlet, Neumann, Neumann, Dirichlet)
Ψ_BCs = BoundaryConditions(mesh, Ψ_bc_fncs, Ψ_bc_types)
Φ_q = (x, y, t, α, β, γ, p) -> (-α[1] / 4, -β[1] / 4)
Ψ_q = (x, y, t, α, β, γ, p) -> (-α[2] / 4, -β[2] / 4)
Φ_S = (x, y, t, (Φ, Ψ), p) -> Φ^2 * Ψ - 2Φ
Ψ_S = (x, y, t, (Φ, Ψ), p) -> -Φ^2 * Ψ + Φ
Φ_exact = (x, y, t) -> exp(-x - y - t / 2)
Ψ_exact = (x, y, t) -> exp(x + y + t / 2)
Φ₀ = [Φ_exact(x, y, 0) for (x, y) in each_point(tri)]
Ψ₀ = [Ψ_exact(x, y, 0) for (x, y) in each_point(tri)]
Φ_prob = FVMProblem(mesh, Φ_BCs; flux_function=Φ_q, source_function=Φ_S,
    initial_condition=Φ₀, final_time=1.0)
Ψ_prob = FVMProblem(mesh, Ψ_BCs; flux_function=Ψ_q, source_function=Ψ_S,
    initial_condition=Ψ₀, final_time=1.0)
prob = FVMSystem(Φ_prob, Ψ_prob)
#sol = solve(prob, TRBDF2(linsolve=KLUFactorization()), parallel=Val(false))

u = prob.initial_condition
du = zero(u)
t = 0.0
p = FVM.get_fvm_parameters(prob, Val(false))
FVM.fvm_eqs!(du,u,p,t)

@benchmark $FVM.fvm_eqs!($du,$u,$p,$t)