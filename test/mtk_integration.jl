using FiniteVolumeMethod
using DelaunayTriangulation
using Test
using SciMLBase
using ModelingToolkit
using DomainSets
using OrdinaryDiffEq
using Symbolics

@testset "ModelingToolkit Integration" begin
    # Test that FVMDiscretization type is exported and works
    @testset "FVMDiscretization Type" begin
        tri = triangulate_rectangle(0, 1, 0, 1, 5, 5, single_boundary = true)
        mesh = FVMGeometry(tri)

        disc = FVMDiscretization(mesh)
        @test disc isa FVMDiscretization
        @test disc.mesh === mesh
        @test disc.boundary_condition_map === nothing

        # Test with boundary condition map
        bc_map = Dict(1 => :left, 2 => :right)
        disc2 = FVMDiscretization(mesh; boundary_condition_map = bc_map)
        @test disc2.boundary_condition_map === bc_map
    end

    @testset "discretize with PDESystem" begin
        @parameters t x y D
        @variables u(..)
        Dt = Differential(t)
        Dx = Differential(x)
        Dy = Differential(y)

        # Heat equation: du/dt = D * (d²u/dx² + d²u/dy²)
        eq = Dt(u(t, x, y)) ~ D * (Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))))

        # Initial condition
        bcs = [u(0, x, y) ~ sin(π * x) * sin(π * y)]

        # Domain
        domains = [t ∈ Interval(0.0, 0.1)]

        # Create PDESystem
        @named pdesys = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)], [D => 1.0])

        # Create mesh
        tri = triangulate_rectangle(0, 1, 0, 1, 10, 10, single_boundary = true)
        mesh = FVMGeometry(tri)

        # Create discretization
        disc = FVMDiscretization(mesh)

        # Discretize
        prob = SciMLBase.discretize(pdesys, disc)

        @test prob isa ODEProblem
        @test prob.tspan == (0.0, 0.1)

        # Solve
        sol = solve(prob, Tsit5(), saveat = 0.01)
        @test sol.retcode == ReturnCode.Success

        # Check solution decays (heat dissipation)
        @test maximum(abs, sol.u[end]) < maximum(abs, sol.u[1])
    end

    @testset "symbolic_discretize" begin
        @parameters t x y D
        @variables u(..)
        Dt = Differential(t)
        Dx = Differential(x)
        Dy = Differential(y)

        eq = Dt(u(t, x, y)) ~ D * (Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))))
        bcs = [u(0, x, y) ~ 0.0]
        domains = [t ∈ Interval(0.0, 1.0)]

        @named pdesys = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)], [D => 1.0])

        tri = triangulate_rectangle(0, 1, 0, 1, 5, 5, single_boundary = true)
        mesh = FVMGeometry(tri)
        disc = FVMDiscretization(mesh)

        # symbolic_discretize should return FVMProblem
        fvm_prob = SciMLBase.symbolic_discretize(pdesys, disc)
        @test fvm_prob isa FVMProblem
    end

    @testset "Reaction-diffusion equation" begin
        @parameters t x y D r
        @variables u(..)
        Dt = Differential(t)
        Dx = Differential(x)
        Dy = Differential(y)

        # Fisher-KPP equation: du/dt = D*Δu + r*u*(1-u)
        eq = Dt(u(t, x, y)) ~ D * (Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y)))) +
                              r * u(t, x, y) * (1 - u(t, x, y))

        bcs = [u(0, x, y) ~ 0.1]  # Initial condition
        domains = [t ∈ Interval(0.0, 1.0)]

        @named pdesys = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)],
            [D => 0.1, r => 1.0])

        tri = triangulate_rectangle(0, 1, 0, 1, 10, 10, single_boundary = true)
        mesh = FVMGeometry(tri)
        disc = FVMDiscretization(mesh)

        prob = SciMLBase.discretize(pdesys, disc)
        @test prob isa ODEProblem

        sol = solve(prob, Tsit5(), saveat = 0.1)
        @test sol.retcode == ReturnCode.Success

        # For Fisher-KPP, solution should grow towards 1
        @test maximum(sol.u[end]) > maximum(sol.u[1])
    end
end
