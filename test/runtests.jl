using FiniteVolumeMethod
using Test
using SafeTestsets 

@safetestset "FVMGeometry" begin
    include("geometry.jl")
end
@safetestset "BoundaryConditions" begin
    include("boundary_conditions.jl")
end
@safetestset "FVMProblem" begin
    include("problem.jl")
end
@safetestset "FVMEquations" begin
    include("equations.jl")
end
@testset "Example PDEs" begin
    @safetestset "Diffusion equation on a square plate" begin
        include("diffusion_example.jl")
    end
    @safetestset "Diffusion equation on a wedge with mixed BCs" begin
        include("wedge_example.jl")
    end
    @safetestset "Reaction-diffusion equation with a dudt BC on a disk" begin
        include("reaction_example.jl")
    end
    @safetestset "Porous-Medium equation" begin
        include("porous_example.jl")
    end
    @safetestset "Porous-Medium equation with a linear source" begin
        include("porous_linear_example.jl")
    end
    @safetestset "Travelling wave problem" begin
        include("travelling_wave_example.jl")
    end
    @safetestset "Diffusion in annulus" begin
        include("annulus_example.jl")
    end
    @safetestset "Laplace's equation" begin
        include("laplaces_equation.jl")
    end
end
@safetestset "Interpolants" begin
    include("interpolants.jl")
end
@testset "Parallel" begin
    @safetestset "Parallel equations for the diffusion equation on a square plate problem" begin
        include("parallel_diffusion_example.jl")
    end
    @safetestset "Parallel equations for the diffusion equation on a wedge problem" begin
        include("parallel_wedge_example.jl")
    end
    @safetestset "Parallel equations for the reaction-diffusion on a disk problem" begin
        include("parallel_reaction_example.jl")
    end
    @safetestset "Testing the parallel equations for a tissue problem" begin
        include("parallel_tissue_example.jl")
    end
end