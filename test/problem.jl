using ..FiniteVolumeMethod
using Test
using LinearAlgebra
using DelaunayTriangulation
using StructEquality
const FVM = FiniteVolumeMethod
const DT = DelaunayTriangulation
@struct_hash_equal FVM.Conditions
include("test_functions.jl")

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
@test FVM.get_conditions(system, 1) == system.problems[1].conditions
@test FVM.get_conditions(system, 2) == system.problems[2].conditions
@test FVM.get_conditions(system, 3) == system.problems[3].conditions
@test FVM.get_conditions(system, 4) == system.problems[4].conditions
@test FVM.get_conditions(system, 5) == system.problems[5].conditions

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

