@testset "Diffusion on a square plate setup" begin
    x₁ = LinRange(0, 2, 10)
    y₁ = LinRange(0, 0, 10)
    x₂ = LinRange(2, 2, 10)
    y₂ = LinRange(0, 2, 10)
    x₃ = LinRange(2, 0, 10)
    y₃ = LinRange(2, 2, 10)
    x₄ = LinRange(0, 0, 10)
    y₄ = LinRange(2, 0, 10)
    x = reduce(vcat, [x₁, x₂, x₃, x₄])
    y = reduce(vcat, [y₁, y₂, y₃, y₄])
    xy = [[x[i], y[i]] for i in eachindex(x)]
    unique!(xy)
    x = [xy[i][1] for i in eachindex(xy)]
    y = [xy[i][2] for i in eachindex(xy)]
    T, adj, adj2v, DG, pts, BN = generate_mesh(x, y, 0.1; gmsh_path=GMSH_PATH)
    mesh = FVMGeometry(T, adj, adj2v, DG, pts, BN)
    boundary_functions = Vector{Function}([(x, y, t, p) -> 0.0])
    type = ["Dirichlet"]
    boundary_conditions = BoundaryConditions(mesh, boundary_functions, type)
    f = (x, y) -> y ≤ 1.0 ? 50.0 : 0.0
    D = (x, y, t, u, p) -> 1 / p[1]
    R = (x, y, t, u, p) -> 0.0'
    u₀ = [f(pts[:, i]...) for i in axes(pts, 2)]
    final_time = 0.5
    prob = FVMProblem(; mesh, boundary_conditions, diffusion=D, reaction=R, diffusion_parameters=9.0, initial_condition=u₀, final_time)
    for τ in prob.mesh.elements
        for i in 1:9
            @test FVM.gets(prob, τ, i) == prob.mesh.shape_function_coeffs[τ][i]
            if i < 4
                @test FVM.getmidpoint(prob, τ, i) == prob.mesh.midpoints[τ][i]
                @test FVM.getnormal(prob, τ, i) == prob.mesh.normals[τ][i]
                @test FVM.getlength(prob, τ, i) == prob.mesh.lengths[τ][i]
                q1 = zeros(2)
                q2 = zeros(2)
                r1, r2, r3, r4, r5, r6 = rand(6)
                prob.flux!(q1, r1, r2, r3, r4, r5, r6, prob.flux_parameters)
                FVM.getflux!(prob, q2, r1, r2, r3, r4, r5, r6, prob.flux_parameters)
                @test q1 == q2
                r1, r2, r3, r4 = rand(4)
                @test FVM.getreaction(prob, r1, r2, r3, r4, nothing) == R(r1, r2, r3, r4, nothing)
            end
        end
    end
    for i in 1:num_points(prob.mesh.points)
        @test FVM.getxy(prob, i) == prob.mesh.points[:, i]
        @test FVM.getvolume(prob, i) == prob.mesh.volumes[i]
    end
    @test prob.mesh == mesh
    @test prob.boundary_conditions == boundary_conditions
    function flux_fnc_test1!(q, x, y, t, α, β, γ, p)
        p_diffusion = p
        q[1] = -D(x, y, t, α * x + β * y + γ, p_diffusion) * α
        q[2] = -D(x, y, t, α * x + β * y + γ, p_diffusion) * β
        return nothing
    end
    flux_parameters = 9.0
    q = zeros(2)
    flux_fnc_test1!(q, 0.5, 0.3, 0.1, 0.5, 0.2, 0.3, flux_parameters)
    @test prob.flux_parameters == flux_parameters
    new_q = zeros(2)
    prob.flux!(new_q, 0.5, 0.3, 0.1, 0.5, 0.2, 0.3, prob.flux_parameters)
    @test q ≈ new_q
    @test prob.reaction_parameters === nothing
    @test prob.initial_condition == u₀
    @test prob.final_time == 0.5
    @test prob.solver === Tsit5()
    @test !prob.steady
end

@testset "Diffusion on a wedge setup" begin
    α = π / 4
    r₁ = LinRange(0, 1, 100)
    θ₁ = LinRange(0, 0, 100)
    x₁ = @. r₁ * cos(θ₁)
    y₁ = @. r₁ * sin(θ₁)
    r₂ = LinRange(1, 1, 100)
    θ₂ = LinRange(0, α, 100)
    x₂ = @. r₂ * cos(θ₂)
    y₂ = @. r₂ * sin(θ₂)
    r₃ = LinRange(1, 0, 100)
    θ₃ = LinRange(α, α, 100)
    x₃ = @. r₃ * cos(θ₃)
    y₃ = @. r₃ * sin(θ₃)
    x = [x₁, x₂, x₃]
    y = [y₁, y₂, y₃]
    T, adj, adj2v, DG, pts, BN = generate_mesh(x, y, 0.1; gmsh_path=GMSH_PATH)
    mesh = FVMGeometry(T, adj, adj2v, DG, pts, BN)
    boundary_functions = [(x, y, p) -> 0.0, (x, y, t, p) -> 0.0, (x, y, p) -> 0.0]
    type = ["Neumann", "Dirichlet", "Neumann"]
    boundary_conditions = BoundaryConditions(mesh, boundary_functions, type)
    f = (x, y) -> 1 - sqrt(x^2 + y^2)
    D = (x, y, t, u, p) -> 1.0
    R = (x, y, t, u, p) -> 0.0
    u₀ = [f(pts[:, i]...) for i in axes(pts, 2)]
    final_time = 0.1
    prob = FVMProblem(; mesh, boundary_conditions, diffusion=D, reaction=R, initial_condition=u₀, final_time)
    for τ in prob.mesh.elements
        for i in 1:9
            @test FVM.gets(prob, τ, i) == prob.mesh.shape_function_coeffs[τ][i]
            if i < 4
                @test FVM.getmidpoint(prob, τ, i) == prob.mesh.midpoints[τ][i]
                @test FVM.getnormal(prob, τ, i) == prob.mesh.normals[τ][i]
                @test FVM.getlength(prob, τ, i) == prob.mesh.lengths[τ][i]
                q1 = zeros(2)
                q2 = zeros(2)
                r1, r2, r3, r4, r5, r6 = rand(6)
                prob.flux!(q1, r1, r2, r3, r4, r5, r6, prob.flux_parameters)
                FVM.getflux!(prob, q2, r1, r2, r3, r4, r5, r6, prob.flux_parameters)
                @test q1 == q2
                r1, r2, r3, r4 = rand(4)
                @test FVM.getreaction(prob, r1, r2, r3, r4, nothing) == R(r1, r2, r3, r4, nothing)
            end
        end
    end
    @test prob.mesh == mesh
    @test prob.boundary_conditions == boundary_conditions
    function flux_fnc_test2!(q, x, y, t, α, β, γ, p)
        p_diffusion = p
        q[1] = -D(x, y, t, α * x + β * y + γ, p_diffusion) * α
        q[2] = -D(x, y, t, α * x + β * y + γ, p_diffusion) * β
        return nothing
    end
    flux_parameters = nothing
    q = zeros(2)
    flux_fnc_test2!(q, 0.5, 0.3, 0.1, 0.5, 0.2, 0.3, flux_parameters)
    @test prob.flux_parameters == flux_parameters
    new_q = zeros(2)
    prob.flux!(new_q, 0.5, 0.3, 0.1, 0.5, 0.2, 0.3, prob.flux_parameters)
    @test q ≈ new_q
    @test prob.reaction_parameters === nothing
    @test prob.initial_condition == u₀
    @test prob.final_time == 0.1
    @test prob.solver === Tsit5()
    @test !prob.steady
end

@testset "Reaction-diffusion with du/dt condition setup" begin
    r = LinRange(1, 1, 1000)
    θ = LinRange(0, 2π, 1000)
    x = @. r * cos(θ)
    y = @. r * sin(θ)
    T, adj, adj2v, DG, pts, BN = generate_mesh(x, y, 0.1; gmsh_path=GMSH_PATH)
    mesh = FVMGeometry(T, adj, adj2v, DG, pts, BN)
    boundary_functions = Vector{Function}([(x, y, t, u, p) -> u])
    type = ["dudt"]
    boundary_conditions = BoundaryConditions(mesh, boundary_functions, type)
    f = (x, y) -> sqrt(besseli(0.0, sqrt(2) * sqrt(x^2 + y^2)))
    D = (x, y, t, u, p) -> u
    R = (x, y, t, u, p) -> u * (1 - u)
    u₀ = [f(pts[:, i]...) for i in axes(pts, 2)]
    final_time = 0.10
    prob_iip = FVMProblem(; mesh, boundary_conditions, diffusion=D, reaction=R, initial_condition=u₀, final_time)
    prob_non_iip = FVMProblem(; mesh, iip_flux=false, boundary_conditions, diffusion=D, reaction=R, initial_condition=u₀, final_time)
    prob_iip_specified = FVMProblem(; mesh, iip_flux=true, boundary_conditions, diffusion=D, reaction=R, initial_condition=u₀, final_time)
    for (i, prob) in pairs([prob_iip, prob_non_iip, prob_iip_specified])
        if i == 1
            @test SciMLBase.isinplace(prob)
        elseif i == 2
            @test !SciMLBase.isinplace(prob)
        elseif i == 3
            @test SciMLBase.isinplace(prob)
        end
        for k in prob.mesh.elements
            for i in 1:9
                @test FVM.gets(prob, k, i) == prob.mesh.shape_function_coeffs[k][i]
                if i < 4
                    @test FVM.getmidpoint(prob, k, i) == prob.mesh.midpoints[k][i]
                    @test FVM.getnormal(prob, k, i) == prob.mesh.normals[k][i]
                    @test FVM.getlength(prob, k, i) == prob.mesh.lengths[k][i]
                    q1 = zeros(2)
                    q2 = zeros(2)
                    r1, r2, r3, r4, r5, r6 = rand(6)
                    if SciMLBase.isinplace(prob)
                        prob.flux!(q1, r1, r2, r3, r4, r5, r6, prob.flux_parameters)
                        FVM.getflux!(prob, q2, r1, r2, r3, r4, r5, r6, prob.flux_parameters)
                        @test q1 == q2
                    else
                        q1 = prob.flux!(r1, r2, r3, r4, r5, r6, prob.flux_parameters)
                        q2 = FVM.getflux!(prob, q2, r1, r2, r3, r4, r5, r6, prob.flux_parameters)
                        @test q1 == q2
                    end
                    r1, r2, r3, r4 = rand(4)
                    @test FVM.getreaction(prob, r1, r2, r3, r4, nothing) == R(r1, r2, r3, r4, nothing)
                end
            end
        end
        @test prob.mesh == mesh
        @test prob.boundary_conditions == boundary_conditions
        if SciMLBase.isinplace(prob)
            function flux_fnc_test3!(q, x, y, t, α, β, γ, p)
                p_diffusion = p
                q[1] = -D(x, y, t, α * x + β * y + γ, p_diffusion) * α
                q[2] = -D(x, y, t, α * x + β * y + γ, p_diffusion) * β
                return nothing
            end
            flux_parameters = nothing
            q = zeros(2)
            flux_fnc_test3!(q, 0.5, 0.3, 0.1, 0.5, 0.2, 0.3, flux_parameters)
            @test prob.flux_parameters == flux_parameters
            new_q = zeros(2)
            prob.flux!(new_q, 0.5, 0.3, 0.1, 0.5, 0.2, 0.3, prob.flux_parameters)
            @test q ≈ new_q
            new_q = zeros(2)
            FVM.getflux!(prob, new_q, 0.5, 0.3, 0.1, 0.5, 0.2, 0.3, prob.flux_parameters)
            @test new_q ≈ q
        else
            function abcflux_fnc_test3!(x, y, t, α, β, γ, p)
                p_diffusion = p
                q = zeros(2)
                q[1] = -D(x, y, t, α * x + β * y + γ, p_diffusion) * α
                q[2] = -D(x, y, t, α * x + β * y + γ, p_diffusion) * β
                return q
            end
            flux_parameters = nothing
            q = abcflux_fnc_test3!(0.5, 0.3, 0.1, 0.5, 0.2, 0.3, flux_parameters)
            @test prob.flux_parameters == flux_parameters
            new_q = prob.flux!(0.5, 0.3, 0.1, 0.5, 0.2, 0.3, prob.flux_parameters)
            @test q ≈ new_q
            new_q = FVM.getflux!(prob, new_q, 0.5, 0.3, 0.1, 0.5, 0.2, 0.3, prob.flux_parameters)
            @test new_q ≈ q
        end
        @test prob.reaction_parameters === nothing
        @test prob.initial_condition == u₀
        @test prob.final_time == 0.1
        @test prob.solver === Tsit5()
        @test !prob.steady
    end
end

@testset "Travelling wave problem setup" begin
    a, b, c, d, Nx, Ny = 0.0, 3.0, 0.0, 40.0, 20, 30
    Tr,adj,adj2v,DG,pts, BN = triangulate_structured(a, b, c, d, Nx, Ny; return_boundary_types=true)
    mesh = FVMGeometry(Tr,adj,adj2v,DG,pts, BN)
    a₁ = (x, y, t, p) -> 1.0
    a₂ = (x, y, t, u, p) -> 0.0
    a₃ = (x, y, t, p) -> 0.0
    a₄ = (x, y, t, u, p) -> 0.0
    type = [:D, :N, :D, :N]
    boundary_conditions = BoundaryConditions(mesh, [a₁, a₂, a₃, a₄], type)
    f = (x, y) -> 0.0
    R = (x, y, t, u, p) -> p * u * (1 - u)
    T = (t, p) -> 1.0
    D = (x, y, t, u, p) -> p * u
    u₀ = [f(pts[:, i]...) for i in axes(pts,2)]
    final_time = 50.0
    diffus = 0.9
    prolif = 0.99
    prob = FVMProblem(; mesh, boundary_conditions, diffusion=D, reaction=R, delay=T, initial_condition=u₀, final_time, reaction_parameters=prolif, diffusion_parameters=diffus)
    for k in Tr
        for i in 1:9
            @test FVM.gets(prob, k, i) == prob.mesh.shape_function_coeffs[k][i]
            if i < 4
                @test FVM.getmidpoint(prob, k, i) == prob.mesh.midpoints[k][i]
                @test FVM.getnormal(prob, k, i) == prob.mesh.normals[k][i]
                @test FVM.getlength(prob, k, i) == prob.mesh.lengths[k][i]
                q1 = zeros(2)
                q2 = zeros(2)
                r1, r2, r3, r4, r5, r6 = rand(6)
                prob.flux!(q1, r1, r2, r3, r4, r5, r6, prob.flux_parameters)
                FVM.getflux!(prob, q2, r1, r2, r3, r4, r5, r6, prob.flux_parameters)
                @test q1 == q2
                r1, r2, r3, r4 = rand(4)
                @test FVM.getreaction(prob, r1, r2, r3, r4, (nothing, prolif, T, R)) == R(r1, r2, r3, r4, prolif) == FVM.getreaction(prob, r1, r2, r3, r4, prob.reaction_parameters)
            end
        end
    end
    @test prob.mesh == mesh
    @test prob.boundary_conditions == boundary_conditions
    function flux_fnc_test4!(q, x, y, t, α, β, γ, p)
        p_delay, p_diffusion = p
        q[1] = -T(t, p_delay) * D(x, y, t, α * x + β * y + γ, p_diffusion) * α
        q[2] = -T(t, p_delay) * D(x, y, t, α * x + β * y + γ, p_diffusion) * β
        return nothing
    end
    flux_parameters = [nothing, diffus]
    q = zeros(2)
    flux_fnc_test4!(q, 0.5, 0.3, 0.1, 0.5, 0.2, 0.3, flux_parameters)
    @test prob.flux_parameters[1:2] == flux_parameters[1:2]
    new_q = zeros(2)
    prob.flux!(new_q, 0.5, 0.3, 0.1, 0.5, 0.2, 0.3, prob.flux_parameters)
    @test q ≈ new_q
    @test prob.reaction_parameters == [nothing, prolif]
    @test prob.initial_condition == u₀
    @test prob.final_time == 50.0
    @test prob.solver === Tsit5()
    @test !prob.steady
end
