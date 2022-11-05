@testset "Basic construction" begin
    α = 3.0
    β = 3.7
    γ = 2.9
    T = (1, 2, 30)
    u₁ = 1.0
    u₂ = 2.71
    u₃ = 2.91
    interp = FVMInterpolant(α, β, γ, T, (0.0, 0.0), (1.0, 2.0), (2.7, 3.0), u₁, u₂, u₃)
    @test interp == FVMInterpolant{Float64,NTuple{2,Float64}}(α, β, γ, T, (0.0, 0.0), (1.0, 2.0), (2.7, 3.0), u₁, u₂, u₃)
    @test interp(2.71, 5.0) ≈ 29.53
    @test interp(0.0, 0.0) ≈ γ
    @test interp(0.0, 1.0) ≈ β + γ
    @test interp(1.0, 0.0) ≈ α + γ
    @test interp((2.71, 5.0)) ≈ 29.53
    @test eval_interpolant(interp, 2.71, 5.0) ≈ 29.53
    @test FVM.nodal_values(interp) == (1.0, 2.71, 2.91)
    @test points(interp) == ((0.0, 0.0), (1.0, 2.0), (2.7, 3.0))
    @test FVM.utype(interp) == Float64

    α = 2
    β = 4.71
    γ = 5
    T = (1, 2, 3)
    p = [[0.3333330, 0.0], [1.0, 2.0], [2.7, 3.0]]
    u₁ = 1.04515
    u₂ = 562.71
    u₃ = 442.9232321
    interp = FVMInterpolant(α, β, γ, T, p..., u₁, u₂, u₃)
    @test interp == FVMInterpolant{Float64,Vector{Float64}}(float(α), β, float(γ), T, p..., u₁, u₂, u₃)
    @test interp(1.7, 1) ≈ 13.11
    @test interp(-5.0, 0.0) ≈ -5.0
    @test interp(0.0, 1.0) ≈ β + γ
    @test interp(1.0, 0.0) ≈ α + γ
    @test interp((1.7, 1)) ≈ 13.11
    @test eval_interpolant(interp, (-5.0, 0.0)) ≈ -5.0
    @test eval_interpolant(interp, 0.0, 1.0) ≈ β + γ
    @test FVM.nodal_values(interp) == (u₁, u₂, u₃)
    @test points(interp) == Tuple(p)
    @test FVM.utype(interp) == Float64

    αβγ = [3.71, 2.0, 5]
    T = (1, 2, 3)
    p = [[0.3333330, 0.0], [1.0, 2.0], [2.7, 3.0]]
    u = [1.0, 3.0, 4.0]
    interp = FVMInterpolant(αβγ, T, p, u)
    @test interp == FVMInterpolant{Float64,Vector{Float64}}(αβγ[1], αβγ[2], αβγ[3], T, p..., u[1], u[2], u[3])
    @test interp(1.7, 3.71) ≈ 18.727
    @test interp(1.0, 0.0) ≈ 8.71
    @test interp(0.0, 1.0) ≈ 7.0
    @test interp(0.0, 0.0) ≈ 5.0
    @test interp((0.0, 0.0)) ≈ 5.0
    @test interp((1.7, 3.71)) ≈ 18.727
    @test eval_interpolant(interp, [1.7, 3.71]) ≈ 18.727
    @test eval_interpolant(interp, 0.0, 1.0) ≈ 7.0
    @test FVM.nodal_values(interp) == (1.0, 3.0, 4.0)
    @test points(interp) == Tuple(p)
    @test FVM.utype(interp) == Float64
end

@testset "Constructing interpolants from FVM solutions" begin
    for problem in [DiffusionEquationOnASquarePlate, ReactionDiffusiondudt, DiffusionOnAWedge, TravellingWaveProblem]
        prob, _ = problem()
        sol = solve(prob, prob.solver)
        test_u = sol(0.2)
        mesh = prob.mesh
        @test_throws MethodError construct_mesh_interpolant(mesh, [])
        interpolants = construct_mesh_interpolant(mesh, test_u)
        @test FVM.utype(interpolants) == Float64
        for k in mesh.elements
            v1, v2, v3 = indices(k)
            p1, p2, p3 = _get_point(mesh.points, v1), _get_point(mesh.points, v2), _get_point(mesh.points, v3)
            @test interpolants[k](p1) ≈ test_u[v1] atol = 1e-7
            @test interpolants[k](p2) ≈ test_u[v2] atol = 1e-7
            @test interpolants[k](p3) ≈ test_u[v3] atol = 1e-7
            @test interpolants[k](p1) ≈ interpolants[k].u₁ atol = 1e-7
            @test interpolants[k](p2) ≈ interpolants[k].u₂ atol = 1e-7
            @test interpolants[k](p3) ≈ interpolants[k].u₃ atol = 1e-7
            @test interpolants[k].tr == k
            @test FVM.nodal_values(interpolants, k) == FVM.nodal_values(interpolants[k]) == (interpolants[k].u₁, interpolants[k].u₂, interpolants[k].u₃)
            @test points(interpolants, k) == points(interpolants[k]) == (p1, p2, p3)
        end
        cmi1 = construct_mesh_interpolant(mesh, test_u)
        cmi2 = construct_mesh_interpolant(mesh, sol, 0.2)
        for (T, I) in cmi1
            for nm in propertynames(I)
                @test getproperty(cmi1[T], nm) == getproperty(cmi2[T], nm)
            end
        end
        _interpolants = deepcopy(interpolants)
        FVM.construct_mesh_interpolant!(interpolants, mesh, test_u)
        for (T, I) in interpolants
            for nm in propertynames(I)
                @test getproperty(interpolants[T], nm) == getproperty(_interpolants[T], nm)
            end
        end
        @test FVM.utype(_interpolants) == Float64

        times = [0.1, 0.2, 0.3, 0.4, 0.5]
        all_interpolants = construct_mesh_interpolant(mesh, sol, times)
        @test FVM.utype(all_interpolants) == Float64
        @test all_interpolants isa NTuple{5,Dict{NTuple{3,Int64},FVMInterpolant{Float64,Vector{Float64}}}}
        for (i, interpolant) in pairs(all_interpolants)
            u = sol(times[i])
            for k in mesh.elements
                v1, v2, v3 = indices(k)
                p1, p2, p3 = _get_point(mesh.points, v1), _get_point(mesh.points, v2), _get_point(mesh.points, v3)
                @test interpolant[k](p1) ≈ u[v1] atol = 1e-7
                @test interpolant[k](p2) ≈ u[v2] atol = 1e-7
                @test interpolant[k](p3) ≈ u[v3] atol = 1e-7
                @test interpolants[k](p1) ≈ interpolants[k].u₁ atol = 1e-7
                @test interpolants[k](p2) ≈ interpolants[k].u₂ atol = 1e-7
                @test interpolants[k](p3) ≈ interpolants[k].u₃ atol = 1e-7
                @test interpolants[k].tr == k
                @test FVM.nodal_values(interpolants, k) == FVM.nodal_values(interpolants[k]) == (interpolants[k].u₁, interpolants[k].u₂, interpolants[k].u₃)
                @test points(interpolants, k) == points(interpolants[k]) == (p1, p2, p3)
            end
        end
        _all_interpolants = deepcopy(all_interpolants)
        FVM.construct_mesh_interpolant!(all_interpolants, mesh, sol, times)
        for (T, I) in interpolants
            for nm in propertynames(I)
                @test getproperty(interpolants[T], nm) == getproperty(_interpolants[T], nm)
            end
        end

        all_interpolants = construct_mesh_interpolant(mesh, sol)
        @test all_interpolants isa NTuple{length(sol.t),Dict{NTuple{3,Int64},FVMInterpolant{Float64,Vector{Float64}}}}
        for (i, interpolant) in pairs(all_interpolants)
            u = sol.u[i]
            for k in mesh.elements
                v1, v2, v3 = indices(k)
                p1, p2, p3 = _get_point(mesh.points, v1), _get_point(mesh.points, v2), _get_point(mesh.points, v3)
                @test interpolant[k](p1) ≈ u[v1] atol = 1e-7
                @test interpolant[k](p2) ≈ u[v2] atol = 1e-7
                @test interpolant[k](p3) ≈ u[v3] atol = 1e-7
                @test interpolants[k](p1) ≈ interpolants[k].u₁ atol = 1e-7
                @test interpolants[k](p2) ≈ interpolants[k].u₂ atol = 1e-7
                @test interpolants[k](p3) ≈ interpolants[k].u₃ atol = 1e-7
                @test interpolants[k].tr == k
                @test FVM.nodal_values(interpolants, k) == FVM.nodal_values(interpolants[k]) == (interpolants[k].u₁, interpolants[k].u₂, interpolants[k].u₃)
                @test points(interpolants, k) == points(interpolants[k]) == (p1, p2, p3)
            end
        end

        _all_interpolants = deepcopy(all_interpolants)
        FVM.construct_mesh_interpolant!(all_interpolants, mesh, sol)
        for (T, I) in interpolants
            for nm in propertynames(I)
                @test getproperty(interpolants[T], nm) == getproperty(_interpolants[T], nm)
            end
        end
    end
end
