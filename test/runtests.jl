using FiniteVolumeMethod
using Test 
using SafeTestsets

@safetestset "Geometry" begin
    include("geometry.jl")
end
@safetestset "Conditions" begin
    include("conditions.jl")
end
@safetestset "Problem" begin
    include("problem.jl")
end
@safetestset "Equations" begin
    include("equations.jl")
end
@safetestset "README" begin
    include("README.jl")
end