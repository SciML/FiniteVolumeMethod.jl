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
file_names = [
    "diffusion_equation_in_a_wedge_with_mixed_boundary_conditions.jl",
    "diffusion_equation_on_a_square_plate.jl",
    "diffusion_equation_on_an_annulus.jl",
    "equilibrium_temperature_distribution_with_mixed_boundary_conditions_and_using_ensembleproblems.jl",
    "helmholtz_equation_with_inhomogeneous_boundary_conditions.jl",
    "laplaces_equation_with_internal_dirichlet_conditions.jl",
    "mean_exit_time.jl",
    "piecewise_linear_and_natural_neighbour_interpolation_for_an_advection_diffusion_equation.jl",
    "porous_fisher_equation_and_travelling_waves.jl",
    "porous_medium_equation.jl",
    "reaction_diffusion_brusselator_system_of_pdes.jl",
    "reaction_diffusion_equation_with_a_time_dependent_dirichlet_boundary_condition_on_a_disk.jl",
    "solving_mazes_with_laplaces_equation.jl"
] # do it manually just to make it easier for testing individual files rather than in a loop, e.g. one like 
#=
for file in files
    @testset "Example: $file" begin
        safe_include(joinpath(dir, file))
    end
end
=#
@test length(files) == length(file_names) # make sure we didn't miss any 
safe_include(joinpath(dir, file_names[1]))
safe_include(joinpath(dir, file_names[2]))
safe_include(joinpath(dir, file_names[3]))
safe_include(joinpath(dir, file_names[4]))
safe_include(joinpath(dir, file_names[5]))
safe_include(joinpath(dir, file_names[6]))
safe_include(joinpath(dir, file_names[7]))
safe_include(joinpath(dir, file_names[8]))
safe_include(joinpath(dir, file_names[9]))
safe_include(joinpath(dir, file_names[10]))
safe_include(joinpath(dir, file_names[11]))
safe_include(joinpath(dir, file_names[12]))
safe_include(joinpath(dir, file_names[13]))
