using FiniteVolumeMethod
using Test
using Dates

ct() = Dates.format(now(), "HH:MM:SS")
function safe_include(filename; name=filename) # Workaround for not being able to interpolate into SafeTestset test names
    mod = @eval module $(gensym()) end
    @info "[$(ct())] Testing $name"
    @testset "Example: $name" begin
        Base.include(mod, filename)
    end
end

@testset verbose = true "FiniteVolumeMethod.jl" begin
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
    safe_include(joinpath(dir, file_names[1]); name=file_names[1]) # diffusion_equation_in_a_wedge_with_mixed_boundary_conditions
    safe_include(joinpath(dir, file_names[2]); name=file_names[2]) # diffusion_equation_on_a_square_plate
    safe_include(joinpath(dir, file_names[3]); name=file_names[3]) # diffusion_equation_on_an_annulus
    safe_include(joinpath(dir, file_names[4]); name=file_names[4]) # equilibrium_temperature_distribution_with_mixed_boundary_conditions_and_using_ensembleproblems
    safe_include(joinpath(dir, file_names[5]); name=file_names[5]) # helmholtz_equation_with_inhomogeneous_boundary_conditions
    safe_include(joinpath(dir, file_names[6]); name=file_names[6]) # laplaces_equation_with_internal_dirichlet_conditions
    safe_include(joinpath(dir, file_names[7]); name=file_names[7]) # mean_exit_time
    safe_include(joinpath(dir, file_names[8]); name=file_names[8]) # piecewise_linear_and_natural_neighbour_interpolation_for_an_advection_diffusion_equation
    safe_include(joinpath(dir, file_names[9]); name=file_names[9]) # porous_fisher_equation_and_travelling_waves
    safe_include(joinpath(dir, file_names[10]); name=file_names[10]) # porous_medium_equation
    safe_include(joinpath(dir, file_names[11]); name=file_names[11]) # reaction_diffusion_brusselator_system_of_pdes
    safe_include(joinpath(dir, file_names[12]); name=file_names[12]) # reaction_diffusion_equation_with_a_time_dependent_dirichlet_boundary_condition_on_a_disk
    safe_include(joinpath(dir, file_names[13]); name=file_names[13]) # solving_mazes_with_laplaces_equation

    using Aqua
    @testset "Aqua" begin
        Aqua.test_all(FiniteVolumeMethod; ambiguities=false, project_extras=false) # don't care about julia < 1.2
        Aqua.test_ambiguities(FiniteVolumeMethod) # don't pick up Base and Core...
    end
end