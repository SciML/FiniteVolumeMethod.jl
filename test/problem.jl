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
conds = FVM.Conditions(mesh, BCs, ICs)
@inferred FVM.Conditions(mesh, BCs, ICs)
@inferred FVMProblem(mesh, BCs, ICs;
    flux_function,
    flux_parameters,
    source_function,
    source_parameters,
    initial_condition,
    initial_time=2.0,
    final_time=5.0)
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

steady = SteadyFVMProblem(prob)
@inferred SteadyFVMProblem(prob)
@test FVM.eval_flux_function(steady, x, y, t, α, β, γ) == (qx, qy)
@inferred FVM.eval_flux_function(steady, x, y, t, α, β, γ)
@test FVM._neqs(steady) == 0
@test !FVM.is_system(steady)

system = FVMSystem(prob, prob, prob, prob, prob)
@inferred FVMSystem(prob, prob, prob, prob, prob)
_α = ntuple(_ -> α, 5)
_β = ntuple(_ -> β, 5)
_γ = ntuple(_ -> γ, 5)
@test FVM.eval_flux_function(system, x, y, t, _α, _β, _γ) == ntuple(_ -> (qx, qy), 5)
@inferred FVM.eval_flux_function(system, x, y, t, _α, _β, _γ)
@test system.initial_condition == [initial_condition initial_condition initial_condition initial_condition initial_condition]'
@test FVM._neqs(system) == 5
@test FVM.is_system(system)
@test system.initial_time == 2.0
@test system.final_time == 5.0

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

function test_compute_flux(_prob, steady, system, steady_system)
    local prob
    for prob in (_prob, steady, system, steady_system)
        if prob === steady || prob == steady_system
            prob = prob.problem
        end
        if prob isa SteadyFVMProblem
            u = prob.problem.initial_condition
        else
            u = prob.initial_condition
        end
        tri = prob.mesh.triangulation
        for (i, j) in keys(get_boundary_edge_map(tri))
            k = get_adjacent(tri, i, j)
            p, q, r = get_point(tri, i, j, k)
            _i, _j = i, j
            i, j, k = DelaunayTriangulation.contains_triangle(tri, i, j, k)[1]
            props = prob.mesh.triangle_props[(i, j, k)]
            s = props.shape_function_coefficients
            if FVM.is_system(prob)
                α = ntuple(ℓ -> s[1] * u[ℓ, i] + s[2] * u[ℓ, j] + s[3] * u[ℓ, k], FVM._neqs(prob))
                β = ntuple(ℓ -> s[4] * u[ℓ, i] + s[5] * u[ℓ, j] + s[6] * u[ℓ, k], FVM._neqs(prob))
                γ = ntuple(ℓ -> s[7] * u[ℓ, i] + s[8] * u[ℓ, j] + s[9] * u[ℓ, k], FVM._neqs(prob))
            else
                α = s[1] * u[i] + s[2] * u[j] + s[3] * u[k]
                β = s[4] * u[i] + s[5] * u[j] + s[6] * u[k]
                γ = s[7] * u[i] + s[8] * u[j] + s[9] * u[k]
            end
            qv = FVM.eval_flux_function(prob, ((p .+ q) ./ 2)..., 2.5, α, β, γ)
            ex, ey = (q .- p) ./ norm(p .- q)
            nx, ny = ey, -ex
            @test DelaunayTriangulation.distance_to_polygon((p .+ q) ./ 2 .+ (nx, ny), get_points(tri), get_boundary_nodes(tri)) < 0.0
            @test DelaunayTriangulation.is_right(DelaunayTriangulation.point_position_relative_to_line(p, q, (p .+ q) ./ 2 .+ (nx, ny)))
            _qv = compute_flux(prob, _i, _j, u, 2.5)
            if !FVM.is_system(prob)
                @test _qv ≈ dot(qv, (nx, ny))
            else
                all_qvs = ntuple(ℓ -> dot(qv[ℓ], (nx, ny)), FVM._neqs(prob))
                @test collect(_qv) ≈ collect(all_qvs)
                @test all_qvs[1] ≈ compute_flux(prob.problems[1], _i, _j, @views(u[1, :]), 2.5)
            end
            @inferred compute_flux(prob, _i, _j, u, 2.5)
        end
        for (i, j) in each_solid_edge(tri)
            if (i, j) ∉ keys(get_boundary_edge_map(tri))
                p, q = get_point(tri, i, j)
                k = get_adjacent(tri, j, i)
                r = get_point(tri, k)
                a, b, c = j, i, k
                a, b, c = DelaunayTriangulation.contains_triangle(tri, a, b, c)[1]
                props = prob.mesh.triangle_props[(a, b, c)]
                s = props.shape_function_coefficients
                if FVM.is_system(prob)
                    α = ntuple(ℓ -> s[1] * u[ℓ, a] + s[2] * u[ℓ, b] + s[3] * u[ℓ, c], FVM._neqs(prob))
                    β = ntuple(ℓ -> s[4] * u[ℓ, a] + s[5] * u[ℓ, b] + s[6] * u[ℓ, c], FVM._neqs(prob))
                    γ = ntuple(ℓ -> s[7] * u[ℓ, a] + s[8] * u[ℓ, b] + s[9] * u[ℓ, c], FVM._neqs(prob))
                else
                    α = s[1] * u[a] + s[2] * u[b] + s[3] * u[c]
                    β = s[4] * u[a] + s[5] * u[b] + s[6] * u[c]
                    γ = s[7] * u[a] + s[8] * u[b] + s[9] * u[c]
                end
                qv = FVM.eval_flux_function(prob, ((p .+ q) ./ 2)..., 2.5, α, β, γ)
                ex, ey = (q .- p) ./ norm(p .- q)
                nx, ny = ey, -ex
                @test DelaunayTriangulation.is_right(DelaunayTriangulation.point_position_relative_to_line(p, q, (p .+ q) ./ 2 .+ (nx, ny)))
                _qv = compute_flux(prob, i, j, u, 2.5)
                if !FVM.is_system(prob)
                    @test _qv ≈ dot(qv, (nx, ny))
                else
                    all_qvs = ntuple(ℓ -> dot(qv[ℓ], (nx, ny)), FVM._neqs(prob))
                    @test collect(_qv) ≈ collect(all_qvs)
                    @test all_qvs[1] ≈ compute_flux(prob.problems[1], i, j, @views(u[1, :]), 2.5)
                end
                @inferred compute_flux(prob, i, j, u, 2.5)
            end
        end
    end
end
test_compute_flux(prob, steady, system, steady_system)