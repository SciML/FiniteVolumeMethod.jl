###########################################################
##
## Example I: Diffusion equation on a square plate 
##
###########################################################
@testset "Diffusion equation on a square plate" begin
    ## Step 1: Generate the mesh 
    a, b, c, d = 0.0, 2.0, 0.0, 2.0
    n = 500
    x₁ = LinRange(a, b, n)
    x₂ = LinRange(b, b, n)
    x₃ = LinRange(b, a, n)
    x₄ = LinRange(a, a, n)
    y₁ = LinRange(c, c, n)
    y₂ = LinRange(c, d, n)
    y₃ = LinRange(d, d, n)
    y₄ = LinRange(d, c, n)
    x = reduce(vcat, [x₁, x₂, x₃, x₄])
    y = reduce(vcat, [y₁, y₂, y₃, y₄])
    xy = [[x[i], y[i]] for i in eachindex(x)]
    unique!(xy)
    x = getx.(xy)
    y = gety.(xy)
    r = 0.03
    (T, adj, adj2v, DG, points), BN = generate_mesh(x, y, r; gmsh_path=GMSH_PATH)
    mesh = FVMGeometry(T, adj, adj2v, DG, points, BN)

    ## Step 2: Define the boundary conditions 
    bc = ((x, y, t, u::T, p) where {T}) -> zero(T)
    type = :Dirichlet # or :D or :dirichlet or "D" or "Dirichlet"
    BCs = BoundaryConditions(mesh, bc, type, BN)

    ## Step 3: Define the actual PDE 
    f = (x, y) -> y ≤ 1.0 ? 50.0 : 0.0 # initial condition 
    D = (x, y, t, u, p) -> 1 / 9 # You could also define flux = (q, x, y, t, α, β, γ, p) -> (q[1] = -α/9; q[2] = -β/9)
    R = ((x, y, t, u::T, p) where {T}) -> zero(T)
    u₀ = @views f.(points[1, :], points[2, :])
    iip_flux = true
    final_time = 0.5
    prob = FVMProblem(mesh, BCs; iip_flux,
        diffusion_function=D, reaction_function=R,
        initial_condition=u₀, final_time)

    ## Step 4: Solve
    alg = TRBDF2(linsolve=KLUFactorization(), autodiff=true)
    sol = solve(prob, alg; specialization=SciMLBase.FullSpecialize, saveat=0.05)

    ## Step 5: Visualisation 
    pt_mat = Matrix(points')
    T_mat = [collect(T)[i][j] for i in 1:length(T), j in 1:3]
    fig = Figure(resolution=(2068.72f0, 686.64f0), fontsize=38)
    ax = Axis(fig[1, 1], width=600, height=600)
    xlims!(ax, a, b)
    ylims!(ax, c, d)
    mesh!(ax, pt_mat, T_mat, color=sol.u[1], colorrange=(0, 50), colormap=:matter)
    ax = Axis(fig[1, 2], width=600, height=600)
    xlims!(ax, a, b)
    ylims!(ax, c, d)
    mesh!(ax, pt_mat, T_mat, color=sol.u[6], colorrange=(0, 50), colormap=:matter)
    ax = Axis(fig[1, 3], width=600, height=600)
    xlims!(ax, a, b)
    ylims!(ax, c, d)
    mesh!(ax, pt_mat, T_mat, color=sol.u[11], colorrange=(0, 50), colormap=:matter)
    SAVE_FIGURE && save("figures/heat_equation_test.png", fig)

    ## Step 6: Define the exact solution for comparison later 
    function diffusion_equation_on_a_square_plate_exact_solution(x, y, t, N, M)
        u_exact = zeros(length(x))
        for j in eachindex(x)
            if t == 0.0
                if y[j] ≤ 1.0
                    u_exact[j] = 50.0
                else
                    u_exact[j] = 0.0
                end
            else
                u_exact[j] = 0.0
                for m = 1:M
                    for n = 1:N
                        u_exact[j] += 200 / π^2 * (1 + (-1)^(m + 1)) * (1 - cos(n * π / 2)) / (m * n) * sin(m * π * x[j] / 2) * sin(n * π * y[j] / 2) * exp(-π^2 / 36 * (m^2 + n^2) * t)
                    end
                end
            end
        end
        return u_exact
    end

    ## Step 7: Compare the results
    sol = solve(prob, alg; saveat=0.1)
    all_errs = [Float64[] for _ in eachindex(sol)]
    u_exact = [diffusion_equation_on_a_square_plate_exact_solution(points[1, :], points[2, :], τ, 200, 200) for τ in sol.t]
    u_fvm = reduce(hcat, sol.u)
    u_exact = reduce(hcat, u_exact)
    errs = reduce(hcat, [100abs.(u - û) / maximum(abs.(u)) for (u, û) in zip(eachcol(u_exact), eachcol(u_fvm))])
    @test all(<(0.15), mean.(eachcol(errs)))
    @test all(<(0.15), median.(eachcol(errs)))
    @test mean(errs) < 0.1
    @test median(errs) < 0.1

    ## Step 8: Visualise the comparison 
    fig = Figure(fontsize=42, resolution=(3469.8997f0, 1466.396f0))
    ax = Axis(fig[1, 1], width=600, height=600, title=L"(a):$ $ Exact solution, $t = %$(sol.t[1])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_exact[:, 1], colorrange=(0, 0.5), colormap=:matter)
    ax = Axis(fig[1, 2], width=600, height=600, title=L"(b):$ $ Exact solution, $t = %$(sol.t[2])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_exact[:, 2], colorrange=(0, 0.5), colormap=:matter)
    ax = Axis(fig[1, 3], width=600, height=600, title=L"(c):$ $ Exact solution, $t = %$(sol.t[3])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_exact[:, 3], colorrange=(0, 0.5), colormap=:matter)
    ax = Axis(fig[1, 4], width=600, height=600, title=L"(d):$ $ Exact solution, $t = %$(sol.t[4])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_exact[:, 4], colorrange=(0, 0.5), colormap=:matter)
    ax = Axis(fig[1, 5], width=600, height=600, title=L"(e):$ $ Exact solution, $t = %$(sol.t[5])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_exact[:, 5], colorrange=(0, 0.5), colormap=:matter)
    ax = Axis(fig[2, 1], width=600, height=600, title=L"(f):$ $ Numerical solution, $t = %$(sol.t[1])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 1], colorrange=(0, 0.5), colormap=:matter)
    ax = Axis(fig[2, 2], width=600, height=600, title=L"(g):$ $ Numerical solution, $t = %$(sol.t[2])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 2], colorrange=(0, 0.5), colormap=:matter)
    ax = Axis(fig[2, 3], width=600, height=600, title=L"(h):$ $ Numerical solution, $t = %$(sol.t[3])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 3], colorrange=(0, 0.5), colormap=:matter)
    ax = Axis(fig[2, 4], width=600, height=600, title=L"(i):$ $ Numerical solution, $t = %$(sol.t[4])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 4], colorrange=(0, 0.5), colormap=:matter)
    ax = Axis(fig[2, 5], width=600, height=600, title=L"(j):$ $ Numerical solution, $t = %$(sol.t[5])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 5], colorrange=(0, 0.5), colormap=:matter)
    SAVE_FIGURE && save("figures/heat_equation_test_error.png", fig)
end

###########################################################
##
## Example II: Diffusion equation on a wedge with mixed BCs
##
###########################################################
@testset "Diffusion equation on a wedge with mixed BCs" begin
    ## Step 1: Generate the mesh 
    n = 500
    α = π / 4

    # The bottom edge 
    r₁ = LinRange(0, 1, n)
    θ₁ = LinRange(0, 0, n)
    x₁ = @. r₁ * cos(θ₁)
    y₁ = @. r₁ * sin(θ₁)

    # Arc 
    r₂ = LinRange(1, 1, n)
    θ₂ = LinRange(0, α, n)
    x₂ = @. r₂ * cos(θ₂)
    y₂ = @. r₂ * sin(θ₂)

    # Upper edge 
    r₃ = LinRange(1, 0, n)
    θ₃ = LinRange(α, α, n)
    x₃ = @. r₃ * cos(θ₃)
    y₃ = @. r₃ * sin(θ₃)

    # Combine and create the mesh 
    x = [x₁, x₂, x₃]
    y = [y₁, y₂, y₃]
    r = 0.01
    tri, BN = generate_mesh(x, y, r; gmsh_path=GMSH_PATH)
    mesh = FVMGeometry(tri, BN)
    T, adj, adj2v, DG, points = tri

    ## Step 2: Define the boundary conditions 
    lower_bc = ((x, y, t, u::T, p) where {T}) -> zero(T)
    arc_bc = ((x, y, t, u::T, p) where {T}) -> zero(T)
    upper_bc = ((x, y, t, u::T, p) where {T}) -> zero(T)
    types = (:N, :D, :N)
    boundary_functions = (lower_bc, arc_bc, upper_bc)
    BCs = BoundaryConditions(mesh, boundary_functions, types, BN)

    ## Step 3: Define the actual PDE  
    f = (x, y) -> 1 - sqrt(x^2 + y^2)
    D = ((x, y, t, u::T, p) where {T}) -> one(T)
    u₀ = f.(points[1, :], points[2, :])
    final_time = 0.1 # Do not need iip_flux = true or R(x, y, t, u, p) = 0, these are defaults 
    prob = FVMProblem(mesh, BCs; diffusion_function=D, initial_condition=u₀, final_time)

    flux = (q, x, y, t, α, β, γ, p) -> (q[1] = -α; q[2] = -β; nothing)
    prob2 = FVMProblem(mesh, BCs; diffusion_function=D, initial_condition=u₀, final_time)

    ## Step 4: Solve
    alg = Rosenbrock23(linsolve=UMFPACKFactorization())
    sol = solve(prob, alg; saveat=0.025)
    sol2 = solve(prob2, alg; saveat=0.025)

    ## Step 5: Visualisation 
    pt_mat = Matrix(points')
    T_mat = [collect(T)[i][j] for i in 1:length(T), j in 1:3]
    fig = Figure(resolution=(2131.8438f0, 684.27f0), fontsize=38)
    ax = Axis(fig[1, 1], width=600, height=600)
    mesh!(ax, pt_mat, T_mat, color=sol.u[1], colorrange=(0, 0.5), colormap=:matter)
    ax = Axis(fig[1, 2], width=600, height=600)
    mesh!(ax, pt_mat, T_mat, color=sol.u[3], colorrange=(0, 0.5), colormap=:matter)
    ax = Axis(fig[1, 3], width=600, height=600)
    mesh!(ax, pt_mat, T_mat, color=sol.u[5], colorrange=(0, 0.5), colormap=:matter)
    SAVE_FIGURE && save("figures/diffusion_equation_wedge_test.png", fig)

    ## Step 6: Define the exact solution for comparison later 
    function diffusion_equation_on_a_wedge_exact_solution(x, y, t, α, N, M)
        f = (r, θ) -> 1.0 - r
        ## Compute the ζ: ζ[m, n+1] is the mth zero of the (nπ/α)th order Bessel function of the first kind 
        ζ = zeros(M, N + 2)
        for n in 0:(N+1)
            order = n * π / α
            @views ζ[:, n+1] .= approx_besselroots(order, M)
        end
        A = zeros(M, N + 1) # A[m, n+1] is the coefficient Aₙₘ
        for n in 0:N
            order = n * π / α
            for m in 1:M
                integrand = rθ -> f(rθ[2], rθ[1]) * besselj(order, ζ[m, n+1] * rθ[2]) * cos(order * rθ[1]) * rθ[2]
                A[m, n+1] = 4.0 / (α * besselj(order + 1, ζ[m, n+1])^2) * hcubature(integrand, [0.0, 0.0], [α, 1.0]; abstol=1e-8)[1]
            end
        end
        r = @. sqrt(x^2 + y^2)
        θ = @. atan(y, x)
        u_exact = zeros(length(x))
        for i in 1:length(x)
            for m = 1:M
                u_exact[i] = u_exact[i] + 0.5 * A[m, 1] * exp(-ζ[m, 1]^2 * t) * besselj(0.0, ζ[m, 1] * r[i])
            end
            for n = 1:N
                order = n * π / α
                for m = 1:M
                    u_exact[i] = u_exact[i] + A[m, n+1] * exp(-ζ[m, n+1]^2 * t) * besselj(order, ζ[m, n+1] * r[i]) * cos(order * θ[i])
                end
            end
        end
        return u_exact
    end

    ## Step 7: Compare the results
    all_errs = [Float64[] for _ in eachindex(sol)]
    u_exact = [diffusion_equation_on_a_wedge_exact_solution(points[1, :], points[2, :], τ, α, 22, 24) for τ in sol.t]
    u_fvm = reduce(hcat, sol.u)
    u_exact = reduce(hcat, u_exact)
    _u_exact = deepcopy(u_exact)
    errs = reduce(hcat, [100abs.(u - û) / maximum(abs.(u)) for (u, û) in zip(eachcol(u_exact), eachcol(u_fvm))])
    @test all(<(0.3), mean.(eachcol(errs)))
    @test all(<(0.15), median.(eachcol(errs)))
    @test mean(errs) < 0.15
    @test median(errs) < 0.1

    all_errs2 = [Float64[] for _ in eachindex(sol2)]
    u_exact2 = [diffusion_equation_on_a_wedge_exact_solution(points[1, :], points[2, :], τ, α, 22, 24) for τ in sol2.t]
    u_fvm2 = reduce(hcat, sol2.u)
    u_exact2 = reduce(hcat, u_exact2)
    errs2 = reduce(hcat, [100abs.(u - û) / maximum(abs.(u)) for (u, û) in zip(eachcol(u_exact2), eachcol(u_fvm2))])
    @test errs == errs2
    @test u_fvm2 == u_fvm

    ## Step 8: Visualise the comparison 
    fig = Figure(fontsize=42, resolution=(3586.6597f0, 1466.396f0))
    ax = Axis(fig[1, 1], width=600, height=600, title=L"(a):$ $ Exact solution, $t = %$(sol.t[1])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=_u_exact[:, 1], colorrange=(0, 0.5), colormap=:matter)
    ax = Axis(fig[1, 2], width=600, height=600, title=L"(b):$ $ Exact solution, $t = %$(sol.t[2])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=_u_exact[:, 2], colorrange=(0, 0.5), colormap=:matter)
    ax = Axis(fig[1, 3], width=600, height=600, title=L"(c):$ $ Exact solution, $t = %$(sol.t[3])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=_u_exact[:, 3], colorrange=(0, 0.5), colormap=:matter)
    ax = Axis(fig[1, 4], width=600, height=600, title=L"(d):$ $ Exact solution, $t = %$(sol.t[4])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=_u_exact[:, 4], colorrange=(0, 0.5), colormap=:matter)
    ax = Axis(fig[1, 5], width=600, height=600, title=L"(e):$ $ Exact solution, $t = %$(sol.t[5])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=_u_exact[:, 5], colorrange=(0, 0.5), colormap=:matter)
    ax = Axis(fig[2, 1], width=600, height=600, title=L"(f):$ $ Numerical solution, $t = %$(sol.t[1])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=sol.u[1], colorrange=(0, 0.5), colormap=:matter)
    ax = Axis(fig[2, 2], width=600, height=600, title=L"(g):$ $ Numerical solution, $t = %$(sol.t[2])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=sol.u[2], colorrange=(0, 0.5), colormap=:matter)
    ax = Axis(fig[2, 3], width=600, height=600, title=L"(h):$ $ Numerical solution, $t = %$(sol.t[3])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=sol.u[3], colorrange=(0, 0.5), colormap=:matter)
    ax = Axis(fig[2, 4], width=600, height=600, title=L"(i):$ $ Numerical solution, $t = %$(sol.t[4])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=sol.u[4], colorrange=(0, 0.5), colormap=:matter)
    ax = Axis(fig[2, 5], width=600, height=600, title=L"(j):$ $ Numerical solution, $t = %$(sol.t[5])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=sol.u[5], colorrange=(0, 0.5), colormap=:matter)
    SAVE_FIGURE && save("figures/heat_equation_wedge_test_error.png", fig)
end

###########################################################
##
## Example III: Reaction-diffusion equation with a time-dependent Dirichlet boundary condition on a disk 
##
###########################################################
@testset "Reaction-diffusion equation with a dudt BC on a disk" begin
    ## Step 1: Generate the mesh 
    n = 500
    r = LinRange(1, 1, 1000)
    θ = LinRange(0, 2π, 1000)
    x = @. r * cos(θ)
    y = @. r * sin(θ)
    r = 0.05
    (T, adj, adj2v, DG, points), BN = generate_mesh(x, y, r; gmsh_path=GMSH_PATH)
    mesh = FVMGeometry(T, adj, adj2v, DG, points, BN)

    ## Step 2: Define the boundary conditions 
    bc = (x, y, t, u, p) -> u
    types = :dudt
    BCs = BoundaryConditions(mesh, bc, types, BN)

    ## Step 3: Define the actual PDE  
    f = (x, y) -> sqrt(besseli(0.0, sqrt(2) * sqrt(x^2 + y^2)))
    D = (x, y, t, u, p) -> u
    R = (x, y, t, u, p) -> u * (1 - u)
    u₀ = [f(points[:, i]...) for i in axes(points, 2)]
    final_time = 0.10
    prob = FVMProblem(mesh, BCs; diffusion_function=D, reaction_function=R, initial_condition=u₀, final_time)

    ## Step 4: Solve
    alg = FBDF(linsolve=UMFPACKFactorization())
    sol = solve(prob, alg; saveat=0.025)

    ## Step 5: Visualisation 
    pt_mat = Matrix(points')
    T_mat = [collect(T)[i][j] for i in 1:length(T), j in 1:3]
    fig = Figure(resolution=(2131.8438f0, 684.27f0), fontsize=38)
    ax = Axis(fig[1, 1], width=600, height=600)
    mesh!(ax, pt_mat, T_mat, color=sol.u[1], colorrange=(1, 1.1), colormap=:matter)
    ax = Axis(fig[1, 2], width=600, height=600)
    mesh!(ax, pt_mat, T_mat, color=sol.u[3], colorrange=(1, 1.1), colormap=:matter)
    ax = Axis(fig[1, 3], width=600, height=600)
    mesh!(ax, pt_mat, T_mat, color=sol.u[5], colorrange=(1, 1.1), colormap=:matter)
    SAVE_FIGURE && save("figures/reaction_diffusion_equation_test.png", fig)

    ## Step 6: Define the exact solution for comparison later 
    function reaction_diffusion_exact_solution(x, y, t)
        u_exact = zeros(length(x))
        for i in eachindex(x)
            u_exact[i] = exp(t) * sqrt(besseli(0.0, sqrt(2) * sqrt(x[i]^2 + y[i]^2)))
        end
        return u_exact
    end

    ## Step 7: Compare the results
    all_errs = [Float64[] for _ in eachindex(sol)]
    u_exact = [reaction_diffusion_exact_solution(points[1, :], points[2, :], τ) for τ in sol.t]
    u_fvm = reduce(hcat, sol.u)
    u_exact = reduce(hcat, u_exact)
    errs = reduce(hcat, [100abs.(u - û) / maximum(abs.(u)) for (u, û) in zip(eachcol(u_exact), eachcol(u_fvm))])
    @test all(<(0.1), mean.(eachcol(errs)))
    @test all(<(0.1), median.(eachcol(errs)))
    @test mean(errs) < 0.05
    @test median(errs) < 0.05

    ## Step 8: Visualise the comparison 
    fig = Figure(fontsize=42, resolution=(3586.6597f0, 1466.396f0))
    ax = Axis(fig[1, 1], width=600, height=600, title=L"(a):$ $ Exact solution, $t = %$(sol.t[1])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_exact[:, 1], colorrange=(1, 1.1), colormap=:matter)
    ax = Axis(fig[1, 2], width=600, height=600, title=L"(b):$ $ Exact solution, $t = %$(sol.t[2])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_exact[:, 2], colorrange=(1, 1.1), colormap=:matter)
    ax = Axis(fig[1, 3], width=600, height=600, title=L"(c):$ $ Exact solution, $t = %$(sol.t[3])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_exact[:, 3], colorrange=(1, 1.1), colormap=:matter)
    ax = Axis(fig[1, 4], width=600, height=600, title=L"(d):$ $ Exact solution, $t = %$(sol.t[4])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_exact[:, 4], colorrange=(1, 1.1), colormap=:matter)
    ax = Axis(fig[1, 5], width=600, height=600, title=L"(e):$ $ Exact solution, $t = %$(sol.t[5])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_exact[:, 5], colorrange=(1, 1.1), colormap=:matter)
    ax = Axis(fig[2, 1], width=600, height=600, title=L"(f):$ $ Numerical solution, $t = %$(sol.t[1])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 1], colorrange=(1, 1.1), colormap=:matter)
    ax = Axis(fig[2, 2], width=600, height=600, title=L"(g):$ $ Numerical solution, $t = %$(sol.t[2])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 2], colorrange=(1, 1.1), colormap=:matter)
    ax = Axis(fig[2, 3], width=600, height=600, title=L"(h):$ $ Numerical solution, $t = %$(sol.t[3])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 3], colorrange=(1, 1.1), colormap=:matter)
    ax = Axis(fig[2, 4], width=600, height=600, title=L"(i):$ $ Numerical solution, $t = %$(sol.t[4])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 4], colorrange=(1, 1.1), colormap=:matter)
    ax = Axis(fig[2, 5], width=600, height=600, title=L"(j):$ $ Numerical solution, $t = %$(sol.t[5])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 5], colorrange=(1, 1.1), colormap=:matter)
    SAVE_FIGURE && save("figures/reaction_heat_equation_test_error.png", fig)
end

###########################################################
##
## Example IV: Porous-Medium equation 
##
###########################################################
@testset "Porous-Medium equation" begin
    ## Step 0: Define all the parameters 
    m = 2
    M = 0.37
    D = 2.53
    final_time = 12.0
    ε = 0.1

    ## Step 1: Define the mesh 
    RmM = 4m / (m - 1) * (M / (4π))^((m - 1) / m)
    L = sqrt(RmM) * (D * final_time)^(1 / (2m))
    n = 500
    x₁ = LinRange(-L, L, n)
    x₂ = LinRange(L, L, n)
    x₃ = LinRange(L, -L, n)
    x₄ = LinRange(-L, -L, n)
    y₁ = LinRange(-L, -L, n)
    y₂ = LinRange(-L, L, n)
    y₃ = LinRange(L, L, n)
    y₄ = LinRange(L, -L, n)
    x = reduce(vcat, [x₁, x₂, x₃, x₄])
    y = reduce(vcat, [y₁, y₂, y₃, y₄])
    xy = [(x, y) for (x, y) in zip(x, y)]
    unique!(xy)
    x = getx.(xy)
    y = gety.(xy)
    r = 0.1
    tri, BN = generate_mesh(x, y, r; gmsh_path=GMSH_PATH)
    mesh = FVMGeometry(tri, BN)
    T, adj, adj2v, DG, points = tri

    ## Step 2: Define the boundary conditions 
    bc = ((x, y, t, u::T, p) where {T}) -> zero(T)
    types = :D
    BCs = BoundaryConditions(mesh, bc, types, BN)

    ## Step 3: Define the actual PDE  
    f = (x, y) -> M * 1 / (ε^2 * π) * exp(-1 / (ε^2) * (x^2 + y^2))
    diff_fnc = (x, y, t, u, p) -> p[1] * u^(p[2] - 1)
    diff_parameters = (D, m)
    u₀ = [f(points[:, i]...) for i in axes(points, 2)]
    prob = FVMProblem(mesh, BCs; diffusion_function=diff_fnc,
        diffusion_parameters=diff_parameters, initial_condition=u₀, final_time)

    ## Step 4: Solve
    alg = TRBDF2(linsolve=KLUFactorization())
    sol = solve(prob, alg; saveat=3.0)

    ## Step 5: Visualisation 
    pt_mat = Matrix(points')
    T_mat = [collect(T)[i][j] for i in 1:length(T), j in 1:3]
    fig = Figure(resolution=(2131.8438f0, 684.27f0), fontsize=38)
    ax = Axis(fig[1, 1], width=600, height=600)
    mesh!(ax, pt_mat, T_mat, color=sol.u[1], colorrange=(0.0, 0.05), colormap=:matter)
    ax = Axis(fig[1, 2], width=600, height=600)
    mesh!(ax, pt_mat, T_mat, color=sol.u[3], colorrange=(0.0, 0.05), colormap=:matter)
    ax = Axis(fig[1, 3], width=600, height=600)
    mesh!(ax, pt_mat, T_mat, color=sol.u[5], colorrange=(0.0, 0.05), colormap=:matter)
    SAVE_FIGURE && save("figures/porous_medium_test.png", fig)

    ## Step 6: Define the exact solution for comparison later 
    function porous_medium_exact_solution(x, y, t, m, M, D)
        u_exact = zeros(length(x))
        RmM = 4m / (m - 1) * (M / (4π))^((m - 1) / m)
        for i in eachindex(x)
            if x[i]^2 + y[i]^2 < RmM * (D * t)^(1 / m)
                u_exact[i] = (D * t)^(-1 / m) * ((M / (4π))^((m - 1) / m) - (m - 1) / (4m) * (x[i]^2 + y[i]^2) * (D * t)^(-1 / m))^(1 / (m - 1))
            else
                u_exact[i] = 0.0
            end
        end
        return u_exact
    end

    ## Step 7: Compare the results
    all_errs = [Float64[] for _ in eachindex(sol)]
    u_exact = [porous_medium_exact_solution(points[1, :], points[2, :], τ, m, M, D) for τ in sol.t]
    u_fvm = reduce(hcat, sol.u)
    u_exact = reduce(hcat, u_exact)
    errs = reduce(hcat, [100abs.(u - û) / maximum(abs.(u)) for (u, û) in zip(eachcol(u_exact[:, 2:end]), eachcol(u_fvm[:, 2:end]))])
    @test all(<(0.1), mean.(eachcol(errs)))
    @test all(<(0.1), median.(eachcol(errs)))
    @test mean(errs) < 0.06
    @test median(errs) < 0.06

    ## Step 8: Visualise the comparison 
    fig = Figure(fontsize=42, resolution=(3586.6597f0, 1466.396f0))
    ax = Axis(fig[1, 1], width=600, height=600, title=L"(a):$ $ Exact solution, $t = %$(sol.t[1])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_exact[:, 1], colorrange=(0, 0.05), colormap=:matter)
    ax = Axis(fig[1, 2], width=600, height=600, title=L"(b):$ $ Exact solution, $t = %$(sol.t[2])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_exact[:, 2], colorrange=(0, 0.05), colormap=:matter)
    ax = Axis(fig[1, 3], width=600, height=600, title=L"(c):$ $ Exact solution, $t = %$(sol.t[3])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_exact[:, 3], colorrange=(0, 0.05), colormap=:matter)
    ax = Axis(fig[1, 4], width=600, height=600, title=L"(d):$ $ Exact solution, $t = %$(sol.t[4])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_exact[:, 4], colorrange=(0, 0.05), colormap=:matter)
    ax = Axis(fig[1, 5], width=600, height=600, title=L"(e):$ $ Exact solution, $t = %$(sol.t[5])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_exact[:, 5], colorrange=(0, 0.05), colormap=:matter)
    ax = Axis(fig[2, 1], width=600, height=600, title=L"(f):$ $ Numerical solution, $t = %$(sol.t[1])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 1], colorrange=(0, 0.05), colormap=:matter)
    ax = Axis(fig[2, 2], width=600, height=600, title=L"(g):$ $ Numerical solution, $t = %$(sol.t[2])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 2], colorrange=(0, 0.05), colormap=:matter)
    ax = Axis(fig[2, 3], width=600, height=600, title=L"(h):$ $ Numerical solution, $t = %$(sol.t[3])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 3], colorrange=(0, 0.05), colormap=:matter)
    ax = Axis(fig[2, 4], width=600, height=600, title=L"(i):$ $ Numerical solution, $t = %$(sol.t[4])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 4], colorrange=(0, 0.05), colormap=:matter)
    ax = Axis(fig[2, 5], width=600, height=600, title=L"(j):$ $ Numerical solution, $t = %$(sol.t[5])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 5], colorrange=(0, 0.05), colormap=:matter)
    SAVE_FIGURE && save("figures/porous_medium_test_error.png", fig)
end

###########################################################
##
## Example V: The Porous-Medium equation with a linear source
##
###########################################################
@testset "Porous-Medium equation with a linear source" begin
    ## Step 0: Define all the parameters 
    m = 3.4
    M = 2.3
    D = 0.581
    λ = 0.2
    final_time = 10.0
    ε = 0.1

    ## Step 1: Define the mesh 
    RmM = 4m / (m - 1) * (M / (4π))^((m - 1) / m)
    L = sqrt(RmM) * (D / (λ * (m - 1)) * (exp(λ * (m - 1) * final_time) - 1))^(1 / (2m))
    n = 500
    x₁ = LinRange(-L, L, n)
    x₂ = LinRange(L, L, n)
    x₃ = LinRange(L, -L, n)
    x₄ = LinRange(-L, -L, n)
    y₁ = LinRange(-L, -L, n)
    y₂ = LinRange(-L, L, n)
    y₃ = LinRange(L, L, n)
    y₄ = LinRange(L, -L, n)
    x = reduce(vcat, [x₁, x₂, x₃, x₄])
    y = reduce(vcat, [y₁, y₂, y₃, y₄])
    xy = [(x, y) for (x, y) in zip(x, y)]
    unique!(xy)
    x = getx.(xy)
    y = gety.(xy)
    r = 0.07
    (T, adj, adj2v, DG, points), BN = generate_mesh(x, y, r; gmsh_path=GMSH_PATH)
    mesh = FVMGeometry(T, adj, adj2v, DG, points, BN)

    ## Step 2: Define the boundary conditions 
    bc = ((x, y, t, u::T, p) where {T}) -> zero(T)
    types = :D
    BCs = BoundaryConditions(mesh, bc, types, BN)

    ## Step 3: Define the exact solution for comparison later 
    function porous_medium_exact_solution(x, y, t, m, M)
        u_exact = zeros(length(x))
        RmM = 4m / (m - 1) * (M / (4π))^((m - 1) / m)
        for i in eachindex(x)
            if x[i]^2 + y[i]^2 < RmM * t^(1 / m)
                u_exact[i] = t^(-1 / m) * ((M / (4π))^((m - 1) / m) - (m - 1) / (4m) * (x[i]^2 + y[i]^2) * t^(-1 / m))^(1 / (m - 1))
            else
                u_exact[i] = 0.0
            end
        end
        return u_exact
    end
    function porous_medium_linear_source_exact_solution(x, y, t, m, M, D, λ)
        return exp(λ * t) * porous_medium_exact_solution(x, y, D / (λ * (m - 1)) * (exp(λ * (m - 1) * t) - 1), m, M)
    end

    ## Step 4: Define the actual PDE  
    f = (x, y) -> M * 1 / (ε^2 * π) * exp(-1 / (ε^2) * (x^2 + y^2))
    diff_fnc = (x, y, t, u, p) -> p[1] * abs(u)^(p[2] - 1)
    reac_fnc = (x, y, t, u, p) -> p[1] * u
    diff_parameters = (D, m)
    react_parameter = λ
    u₀ = [f(points[:, i]...) for i in axes(points, 2)]
    prob = FVMProblem(mesh, BCs; diffusion_function=diff_fnc,
        diffusion_parameters=diff_parameters,
        reaction_function=reac_fnc, reaction_parameters=react_parameter,
        initial_condition=u₀, final_time)

    ## Step 5: Solve
    alg = TRBDF2(linsolve=KLUFactorization())
    sol = solve(prob, alg; saveat=2.5)

    ## Step 6: Visualisation 
    pt_mat = Matrix(points')
    T_mat = [collect(T)[i][j] for i in 1:length(T), j in 1:3]
    fig = Figure(resolution=(2131.8438f0, 684.27f0), fontsize=38)
    ax = Axis(fig[1, 1], width=600, height=600)
    mesh!(ax, pt_mat, T_mat, color=sol.u[1], colorrange=(0.0, 0.5), colormap=:matter)
    ax = Axis(fig[1, 2], width=600, height=600)
    mesh!(ax, pt_mat, T_mat, color=sol.u[3], colorrange=(0.0, 0.5), colormap=:matter)
    ax = Axis(fig[1, 3], width=600, height=600)
    mesh!(ax, pt_mat, T_mat, color=sol.u[5], colorrange=(0.0, 0.5), colormap=:matter)
    SAVE_FIGURE && save("figures/porous_medium_linear_source_test.png", fig)

    ## Step 7: Compare the results
    all_errs = [Float64[] for _ in eachindex(sol)]
    u_exact = [porous_medium_linear_source_exact_solution(points[1, :], points[2, :], τ, m, M, D, λ) for τ in sol.t]
    u_fvm = reduce(hcat, sol.u)
    u_exact = reduce(hcat, u_exact)
    errs = reduce(hcat, [100abs.(u - û) / maximum(abs.(u)) for (u, û) in zip(eachcol(u_exact[:, 2:end]), eachcol(u_fvm[:, 2:end]))])
    @test all(<(0.26), mean.(eachcol(errs)))
    @test all(<(0.17), median.(eachcol(errs)))
    @test mean(errs) < 0.25
    @test median(errs) < 0.05

    ## Step 8: Visualise the comparison 
    fig = Figure(fontsize=42, resolution=(3586.6597f0, 1466.396f0))
    ax = Axis(fig[1, 1], width=600, height=600, title=L"(a):$ $ Exact solution, $t = %$(sol.t[1])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_exact[:, 1], colorrange=(0, 0.05), colormap=:matter)
    ax = Axis(fig[1, 2], width=600, height=600, title=L"(b):$ $ Exact solution, $t = %$(sol.t[2])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_exact[:, 2], colorrange=(0, 0.05), colormap=:matter)
    ax = Axis(fig[1, 3], width=600, height=600, title=L"(c):$ $ Exact solution, $t = %$(sol.t[3])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_exact[:, 3], colorrange=(0, 0.05), colormap=:matter)
    ax = Axis(fig[1, 4], width=600, height=600, title=L"(d):$ $ Exact solution, $t = %$(sol.t[4])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_exact[:, 4], colorrange=(0, 0.05), colormap=:matter)
    ax = Axis(fig[1, 5], width=600, height=600, title=L"(e):$ $ Exact solution, $t = %$(sol.t[5])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_exact[:, 5], colorrange=(0, 0.05), colormap=:matter)
    ax = Axis(fig[2, 1], width=600, height=600, title=L"(f):$ $ Numerical solution, $t = %$(sol.t[1])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 1], colorrange=(0, 0.05), colormap=:matter)
    ax = Axis(fig[2, 2], width=600, height=600, title=L"(g):$ $ Numerical solution, $t = %$(sol.t[2])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 2], colorrange=(0, 0.05), colormap=:matter)
    ax = Axis(fig[2, 3], width=600, height=600, title=L"(h):$ $ Numerical solution, $t = %$(sol.t[3])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 3], colorrange=(0, 0.05), colormap=:matter)
    ax = Axis(fig[2, 4], width=600, height=600, title=L"(i):$ $ Numerical solution, $t = %$(sol.t[4])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 4], colorrange=(0, 0.05), colormap=:matter)
    ax = Axis(fig[2, 5], width=600, height=600, title=L"(j):$ $ Numerical solution, $t = %$(sol.t[5])$", titlealign=:left)
    mesh!(ax, pt_mat, T_mat, color=u_fvm[:, 5], colorrange=(0, 0.05), colormap=:matter)
    SAVE_FIGURE && save("figures/porous_medium_linear_source_test_error.png", fig)
end

###########################################################
##
## Example VI: Travelling wave problem 
##
###########################################################
@testset "Travelling wave problem" begin
    ## Step 1: Define the mesh 
    a, b, c, d, Nx, Ny = 0.0, 3.0, 0.0, 40.0, 60, 80
    tri, BN = triangulate_structured(a, b, c, d, Nx, Ny; return_boundary_types=true)
    mesh = FVMGeometry(tri, BN)
    T, adj, adj2v, DG, points = tri

    ## Step 2: Define the boundary conditions 
    a₁ = ((x, y, t, u::T, p) where {T}) -> one(T)
    a₂ = ((x, y, t, u::T, p) where {T}) -> zero(T)
    a₃ = ((x, y, t, u::T, p) where {T}) -> zero(T)
    a₄ = ((x, y, t, u::T, p) where {T}) -> zero(T)
    bc_fncs = (a₁, a₂, a₃, a₄)
    types = (:D, :N, :D, :N)
    BCs = BoundaryConditions(mesh, bc_fncs, types, BN)

    ## Step 3: Define the actual PDE  
    f = ((x::T, y::T) where {T}) -> zero(T)
    diff_fnc = (x, y, t, u, p) -> p * u
    reac_fnc = (x, y, t, u, p) -> p * u * (1 - u)
    D, λ = 0.9, 0.99
    diff_parameters = D
    reac_parameters = λ
    final_time = 50.0
    u₀ = [f(points[:, i]...) for i in axes(points, 2)]
    prob = FVMProblem(mesh, BCs; diffusion_function=diff_fnc, reaction_function=reac_fnc,
        diffusion_parameters=diff_parameters, reaction_parameters=reac_parameters,
        initial_condition=u₀, final_time)

    ## Step 4: Solve
    alg = TRBDF2(linsolve=KLUFactorization())
    sol = solve(prob, alg; saveat=0.5)

    ## Step 5: Visualisation 
    pt_mat = Matrix(points')
    T_mat = [collect(T)[i][j] for i in 1:length(T), j in 1:3]
    fig = Figure(resolution=(3023.5881f0, 684.27f0), fontsize=38)
    ax = Axis(fig[1, 1], width=600, height=600)
    mesh!(ax, pt_mat, T_mat, color=sol.u[1], colorrange=(0.0, 1.0), colormap=:matter)
    ax = Axis(fig[1, 2], width=600, height=600)
    mesh!(ax, pt_mat, T_mat, color=sol.u[51], colorrange=(0.0, 1.0), colormap=:matter)
    ax = Axis(fig[1, 3], width=600, height=600)
    mesh!(ax, pt_mat, T_mat, color=sol.u[101], colorrange=(0.0, 1.0), colormap=:matter)

    ## Step 6: Put the solutions into a matrix format and test the x-invariance by comparing each column to the middle
    u_mat = [reshape(u, (Nx, Ny)) for u in sol.u]
    all_errs = zeros(length(sol))
    err_cache = zeros((Nx - 1) * Ny)
    for i in eachindex(sol)
        u = u_mat[i]
        ctr = 1
        for j in union(1:((Nx÷2)-1), ((Nx÷2)+1):Nx)
            for k in 1:Ny
                err_cache[ctr] = 100abs(u[j, k] .- u[Nx÷2, k])
                ctr += 1
            end
        end
        all_errs[i] = mean(err_cache)
    end
    @test all(all_errs .< 0.05)

    ## Step 7: Now compare to the exact (travelling wave) solution 
    large_time_idx = findfirst(sol.t .== 10)
    c = sqrt(λ / (2D))
    cₘᵢₙ = sqrt(λ * D / 2)
    zᶜ = 0.0
    exact_solution = ((z::T) where {T}) -> ifelse(z ≤ zᶜ, 1 - exp(cₘᵢₙ * (z - zᶜ)), zero(T))
    err_cache = zeros(Nx * Ny)
    all_errs = zeros(length(sol) - large_time_idx + 1)
    for (i, t_idx) in pairs(large_time_idx:lastindex(sol))
        u = u_mat[t_idx]
        τ = sol.t[t_idx]
        ctr = 1
        for j in 1:Nx
            for k in 1:Ny
                y = c + (k - 1) * (d - c) / (Ny - 1)
                z = y - c * τ
                exact_wave = exact_solution(z)
                err_cache[ctr] = abs(u[j, k] - exact_wave)
                ctr += 1
            end
        end
        all_errs[i] = mean(err_cache)
    end
    @test all(all_errs .< 0.1)

    ## Step 8: Visualise the comparison with the travelling wave
    travelling_wave_values = zeros(Ny, length(sol) - large_time_idx + 1)
    z_vals = zeros(Ny, length(sol) - large_time_idx + 1)
    for (i, t_idx) in pairs(large_time_idx:lastindex(sol))
        u = u_mat[t_idx]
        τ = sol.t[t_idx]
        for k in 1:Ny
            y = c + (k - 1) * (d - c) / (Ny - 1)
            z = y - c * τ
            z_vals[k, i] = z
            travelling_wave_values[k, i] = u[Nx÷2, k]
        end
    end
    exact_z_vals = collect(LinRange(extrema(z_vals)..., 500))
    exact_travelling_wave_values = exact_solution.(exact_z_vals)

    ax = Axis(fig[1, 4], width=900, height=600)
    colors = cgrad(:matter, length(sol) - large_time_idx + 1; categorical=false)
    [lines!(ax, z_vals[:, i], travelling_wave_values[:, i], color=colors[i], linewidth=2) for i in 1:(length(sol)-large_time_idx+1)]
    lines!(ax, exact_z_vals, exact_travelling_wave_values, color=:red, linewidth=4, linestyle=:dash)
    SAVE_FIGURE && save("figures/travelling_wave_problem_test.png", fig)
end