using ..FiniteVolumeMethod
using Test
using LinearAlgebra
using DelaunayTriangulation
const FVM = FiniteVolumeMethod
const DT = DelaunayTriangulation
include("test_functions.jl")

prob = example_diffusion_problem()
u = prob.initial_condition
t = 0.0
test_shape_function_coefficients(prob, u)
test_get_flux(prob, u, t)
test_single_triangle(prob, u, t)
test_source_contribution(prob, u, t)
test_dudt_val(prob, u, t)

