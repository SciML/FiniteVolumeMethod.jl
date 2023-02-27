using FiniteVolumeMethod
using Test
using SafeTestsets

@testset verbose = true "FiniteVolumeMethod" begin
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
    @testset verbose = true "Example PDEs" begin
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
        @safetestset "Diffusion in an annulus" begin
            include("annulus_example.jl")
        end
        @safetestset "Laplace's equation" begin
            include("laplaces_equation.jl")
        end
    end
    @testset verbose = true "MET Problems" begin
        @safetestset "Circle" begin
            include("met_circle.jl")
        end
        @safetestset "Perturbed circle" begin
            include("met_perturbed_circle.jl")
        end
        @safetestset "Ellipse" begin
            include("met_ellipse.jl")
        end
        @safetestset "Perturbed ellipse" begin
            include("met_perturbed_ellipse.jl")
        end
        @safetestset "Annulus" begin
            include("met_annulus.jl")
        end
        @safetestset "Perturbed annulus" begin
            include("met_perturbed_annulus.jl")
        end
    end
    @safetestset "Interpolants" begin
        include("interpolants.jl")
    end
    @testset verbose = true "Parallel" begin
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
end