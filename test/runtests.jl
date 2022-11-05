using FiniteVolumeMethod
using Test
using Random
using LinearAlgebra
using StatsBase
using Setfield
using FastGaussQuadrature
using DelaunayTriangulation
using Cubature
using ElasticArrays
using FunctionWranglers
using Bessels
using PreallocationTools
using ForwardDiff
using BandedMatrices
using SparseArrays
using DifferentialEquations

const GMSH_PATH = "C:/Users/licer/.julia/dev/FiniteVolumeMethod/gmsh-4.9.4-Windows64/gmsh.exe"
const FVM = FiniteVolumeMethod

include("template_functions.jl")

@testset "Examples" begin 
    include("example_pdes.jl")
end
@testset "Geometry" begin
    include("geometry.jl")
end
@testset "Boundary" begin
    include("boundary.jl")
end
@testset "Problem" begin
    include("problem.jl")
end
@testset "Parameters" begin
    include("parameters.jl")
end
@testset "Equations" begin
    include("equations.jl")
end
@testset "Sparsity" begin
    include("sparsity.jl")
end
@testset "Interpolants" begin
    include("interpolants.jl")
end