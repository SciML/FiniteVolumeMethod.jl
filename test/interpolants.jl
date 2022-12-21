@testset "Test linear interpolation evaluation and construction" begin
    ## We test on the porous medium equation with a linear source
    m = 3.4
    M = 2.3
    D = 0.581
    λ = 0.2
    final_time = 10.0
    ε = 0.1
    RmM = 4m / (m - 1) * (M / (4π))^((m - 1) / m)
    L = sqrt(RmM) * (D / (λ * (m - 1)) * (exp(λ * (m - 1) * 1.2final_time) - 1))^(1 / (2m))
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
    bc = ((x, y, t, u::T, p) where {T}) -> zero(T)
    types = :D
    BCs = BoundaryConditions(mesh, bc, types, BN)
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
    alg = TRBDF2(linsolve=KLUFactorization())
    sol = solve(prob, alg; saveat=2.5)

    ## Now test on a single example 
    x = 0.37
    y = 0.58
    t_idx = 5 # t = sol.t[t_idx]
    V = jump_and_march(x, y, prob)
    @test DelaunayTriangulation.isintriangle(get_point(points, V...)..., (x, y)) == 1
    val = eval_interpolant(sol, x, y, t_idx, V)
    @test eval_interpolant(sol, x, y, t_idx, V) == eval_interpolant(sol, x, y, sol.t[t_idx], V)
    @inferred eval_interpolant(sol, x, y, t_idx, V)
    @inferred eval_interpolant(sol, x, y, sol.t[t_idx], V)
    SHOW_WARNTYPE && @code_warntype eval_interpolant(sol, x, y, t_idx, V)
    SHOW_WARNTYPE && @code_warntype eval_interpolant(sol, x, y, sol.t[t_idx], V)
    SHOW_WARNTYPE && @code_warntype eval_interpolant!(zeros(3), prob, x, y, V, sol.u[5])

    ## Make sure the interpolant is actually an interpolant 
    for j in DT._eachindex(FVM.get_points(prob))
        V = jump_and_march(get_point(prob, j)..., prob)
        @test DelaunayTriangulation.isintriangle(get_point(points, V...)..., get_point(prob, j)) == 0
        for t_idx in eachindex(sol)
            val = eval_interpolant(sol, get_point(prob, j)..., t_idx, V)
            @test val ≈ sol.u[t_idx][j] atol = 1e-4
        end
    end

    ## Consider a larger set of points 
    nx = 250
    ny = 250
    grid_x = LinRange(-L + 1e-1, L - 1e-1, nx)
    grid_y = LinRange(-L + 1e-1, L - 1e-1, ny)
    V_mat = Matrix{NTuple{3,Int64}}(undef, nx, ny)
    last_triangle = rand(FVM.get_elements(prob)) # initiate 
    for j in 1:ny
        for i in 1:nx
            # global last_triangle
            V_mat[i, j] = jump_and_march(grid_x[i], grid_y[j], prob; try_points=last_triangle)
            last_triangle = V_mat[i, j]
            @test DelaunayTriangulation.isintriangle(get_point(points, V_mat[i, j]...)..., (grid_x[i], grid_y[j])) ≥ 0
        end
    end

    ## Evaluate the interpolant at each time 
    u_vals = zeros(nx, ny, length(sol))
    for k in eachindex(sol)
        for j in 1:ny
            for i in 1:nx
                V = V_mat[i, j]
                u_vals[i, j, k] = eval_interpolant(sol, grid_x[i], grid_y[j], k, V)
            end
        end
    end

    ## Plot 
    fig = Figure(resolution=(2744.0f0, 692.0f0))
    for k in 1:4
        ax = Axis3(fig[1, k])
        zlims!(ax, 0, 1), xlims!(ax, -L - 1e-1, L + 1e-1), ylims!(ax, -L - 1e-1, L + 1e-1)
        surface!(ax, grid_x, grid_y, u_vals[:, :, k+1], colormap=:matter)
    end
    SAVE_FIGURE && save("figures/surface_plots_travelling_wave.png", fig)
end