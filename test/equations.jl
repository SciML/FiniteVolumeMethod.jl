using ..FiniteVolumeMethod
using Test
using PolygonOps
using LinearAlgebra
using DelaunayTriangulation
using OrdinaryDiffEq
using LinearSolve
const FVM = FiniteVolumeMethod
const DT = DelaunayTriangulation
include("test_functions.jl")

@testset "Diffusion problem" begin
    prob = example_diffusion_problem()
    u = prob.initial_condition
    t = 0.0
    test_shape_function_coefficients(prob, u)
    test_get_flux(prob, u, t)
    test_get_boundary_flux(prob, u, t)
    test_single_triangle(prob, u, t)
    test_source_contribution(prob, u, t)
    test_dudt_val(prob, u, t)
end

@testset "Convection problem" begin
    prob = example_heat_convection_problem()
    u = prob.initial_condition
    t = 0.0
    test_shape_function_coefficients(prob, u)
    test_get_flux(prob, u, t)
    test_get_boundary_flux(prob, u, t, false)
    test_single_triangle(prob, u, t)
    test_source_contribution(prob, u, t)
    test_dudt_val(prob, u, t, false)
end

const fvm_eqs! = (du, u, p, t) -> begin
    du1 = zero(du)
    du2 = zero(du)
    FVM.fvm_eqs_flux!(du1, u, p, t)
    FVM.fvm_eqs_source!(du2, u, p, t)
    du .= du1 .+ du2
    return du
end
@testset "Exact test for a corner point: Can we get the Neumann edge contributions correct?" begin
    prob = example_heat_convection_problem()
    u = prob.initial_condition
    t = 0.0
    tri = prob.mesh.triangulation
    i, j, k = 1, 2, 201
    T = i, j, k
    p, q, r = get_point(tri, i, j, k)
    x1 = q[1]
    y1 = r[2]
    u1, u2, u3 = u[[T...]]
    A = PolygonOps.area(((0.0, 0.0), (x1 / 2, 0.0), (x1 / 3, y1 / 3), (0, y1 / 2), (0.0, 0.0)))
    @test prob.mesh.cv_volumes[1] ≈ A
    @test FVM.get_shape_function_coefficients(prob.mesh.triangle_props[T], T, u, prob) == (0.0, 0.0, 10.0)
    props = prob.mesh.triangle_props[T]
    c1, c2, c3, c4 = (0.0, 0.0), (x1 / 2, 0.0), (x1 / 3, y1 / 3), (0.0, y1 / 2)
    m1 = (c1 .+ c2) ./ 2
    m2 = (c2 .+ c3) ./ 2
    m3 = (c3 .+ c4) ./ 2
    m4 = (c4 .+ c1) ./ 2
    ℓ1 = norm(c2 .- c1)
    ℓ2 = norm(c3 .- c2)
    ℓ3 = norm(c4 .- c3)
    ℓ4 = norm(c1 .- c4)
    n2 = let r = c3 .- c2
        rx, ry = r
        ry, -rx
    end
    n3 = let r = c4 .- c3
        rx, ry = r
        ry, -rx
    end
    α, β, γ = 0.0, 0.0, 10.0
    flux1 = prob.conditions.functions[1](m1..., t, α * m1[1] + β * m2[1] + γ) * ℓ1
    flux2 = prob.flux_function(m2..., 0.0, α, β, γ, prob.flux_parameters) |> q -> dot(q, n2)
    flux3 = prob.flux_function(m3..., 0.0, α, β, γ, prob.flux_parameters) |> q -> dot(q, n3)
    flux4 = prob.conditions.functions[4](m4..., t, α * m4[1] + β * m4[2] + γ) * ℓ4
    fl = -(1 / A) * sum((flux1, flux2, flux3, flux4))
    fltest = get_dudt_val(prob, u, t, i, false)
    @test fl ≈ fltest
    flpar = fvm_eqs!(zeros(num_points(tri)), u, FVM.get_fvm_parameters(prob, Val(false)), t)[i]
    flser = fvm_eqs!(zeros(num_points(tri)), u, FVM.get_fvm_parameters(prob, Val(true)), t)[i]
    @test fl ≈ flpar
    @test fl ≈ flser
end


@testset "Diffusion problem" begin
    prob = example_diffusion_problem()
    u = prob.initial_condition
    t = 0.0
    test_shape_function_coefficients(prob, u)
    test_get_flux(prob, u, t)
    test_get_boundary_flux(prob, u, t)
    test_single_triangle(prob, u, t)
    test_source_contribution(prob, u, t)
    test_dudt_val(prob, u, t)
end

@testset "FVMSystem" begin
    prob = example_diffusion_problem()
    @test_throws FVM.InvalidFluxError FVMSystem(prob, prob)
    sys, prob = example_diffusion_problem_system()
    solsys = solve(sys, TRBDF2(linsolve=KLUFactorization()), saveat=0.05)
    solsysser = solve(sys, TRBDF2(linsolve=KLUFactorization()), saveat=0.05, parallel=Val(false))
    solprob = solve(prob, TRBDF2(linsolve=KLUFactorization()), saveat=0.05)
    solprobser = solve(prob, TRBDF2(linsolve=KLUFactorization()), saveat=0.05, parallel=Val(false))
    solprobu = reduce(hcat, solprob.u)
    solprobseru = reduce(hcat, solprobser.u)
    solsysu1 = reduce(hcat, [solsys.u[i][1, :] for i in eachindex(solsys)])
    solsysu2 = reduce(hcat, [solsys.u[i][2, :] for i in eachindex(solsys)])
    solsysser1 = reduce(hcat, [solsysser.u[i][1, :] for i in eachindex(solsysser)])
    solsysser2 = reduce(hcat, [solsysser.u[i][2, :] for i in eachindex(solsysser)])
    @test all(≈(solprobu), (solprobseru, solsysu1, solsysu2, solsysser1, solsysser2))
end