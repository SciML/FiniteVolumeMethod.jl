using FiniteVolumeMethod
using Test 
using SafeTestsets

@safetestset "Geometry" begin
    include("geometry.jl")
end