using FiniteVolumeMethod
using Test
using Dates
ct() = Dates.format(now(), "HH:MM:SS")
function safe_include(filename) # Workaround for not being able to interpolate into SafeTestset test names
    mod = @eval module $(gensym()) end
    @info "[$(ct())] Testing $filename"
    return Base.include(mod, filename)
end

@testset "Geometry" begin
    safe_include("geometry.jl")
end
@testset "Conditions" begin
    safe_include("conditions.jl")
end
@testset "Problem" begin
    safe_include("problem.jl")
end
@testset "Equations" begin
    safe_include("equations.jl")
end
@testset "README" begin
    safe_include("README.jl")
end

dir = joinpath(dirname(@__DIR__), "docs", "src", "literate_tutorials")
files = readdir(dir)
for file in files
    @testset "Example: $file" begin
        safe_include(joinpath(dir, file))
    end
end
