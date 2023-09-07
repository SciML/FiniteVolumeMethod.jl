using ..FiniteVolumeMethod
using Test
using PolygonOps
using LinearAlgebra
using DelaunayTriangulation
const FVM = FiniteVolumeMethod
const DT = DelaunayTriangulation
include("test_functions.jl")

@testset "Diffusion problem" begin
    prob = example_diffusion_problem()
    u = prob.initial_condition
    t = 0.0
    test_shape_function_coefficients(prob, u)
    test_get_flux(prob, u, t)
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
    test_single_triangle(prob, u, t)
    test_source_contribution(prob, u, t)
    test_dudt_val(prob, u, t; is_diff=false)
end

prob = example_heat_convection_problem()
u = prob.initial_condition
t = 0.0
tri = prob.mesh.triangulation
T = i, j, k
p, q, r = get_point(tri, i, j, k)
x1 = q[1]
y1 = r[2]
u1, u2, u3 = u[[T...]]
A = PolygonOps.area(((0.0, 0.0), (x1 / 2, 0.0), (x1 / 3, y1 / 3), (0, y1 / 2), (0.0, 0.0)))
@test prob.mesh.cv_volumes[1] ≈ A
@test FVM.get_shape_function_coefficients(prob.mesh.triangle_props[T], T, u, prob) == (0.0, 0.0, 10.0)
props = prob.mesh.triangle_props[T]
x1, y1, nx1, ny1, ℓ1 = FVM._get_cv_components(props, 1)
x2, y2, nx2, ny2, ℓ2 = FVM._get_cv_components(props, 2)
x3, y3, nx3, ny3, ℓ3 = FVM._get_cv_components(props, 3)
c1, c2, c3, c4 = (0.0, 0.0), (x1 / 2, 0.0), (x1 / 3, y1 / 3), (0.0, y1 / 2)
(c1 .+ c2) ./ 2
x1, y1
α, β, γ = 0.0, 0.0, 10.0