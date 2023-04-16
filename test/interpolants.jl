using ..FiniteVolumeMethod
include("test_setup.jl")
using Test
using CairoMakie
using OrdinaryDiffEq
using LinearSolve
using StableRNGs
using ReferenceTests

## We test on the porous medium equation with a linear source
for parallel in (false, true)
    m = 3.4
    M = 2.3
    D = 0.581
    λ = 0.2
    final_time = 10.0
    ε = 0.1
    RmM = 4m / (m - 1) * (M / (4π))^((m - 1) / m)
    L = sqrt(RmM) * (D / (λ * (m - 1)) * (exp(λ * (m - 1) * final_time) - 1))^(1 / (2m))
    x = [-L, L, L, -L, -L]
    y = [-L, -L, L, L, -L]
    boundary_nodes, points = convert_boundary_points_to_indices(x, y)
    rng = StableRNG(123)
    tri = triangulate(points; boundary_nodes, rng)
    A = get_total_area(tri)
    refine!(tri; max_area=1e-4A / 2, rng)
    mesh = FVMGeometry(tri)
    points = get_points(tri)
    bc = ((x, y, t, u::T, p) where {T}) -> zero(T)
    types = :D
    BCs = BoundaryConditions(mesh, bc, types)
    f = (x, y) -> M * 1 / (ε^2 * π) * exp(-1 / (ε^2) * (x^2 + y^2))
    diff_fnc = (x, y, t, u, p) -> p[1] * abs(u)^(p[2] - 1)
    reac_fnc = (x, y, t, u, p) -> p[1] * u
    diff_parameters = (D, m)
    react_parameter = λ
    u₀ = f.(first.(points), last.(points))
    prob = FVMProblem(mesh, BCs; diffusion_function=diff_fnc,
        diffusion_parameters=diff_parameters,
        reaction_function=reac_fnc, reaction_parameters=react_parameter,
        initial_condition=u₀, final_time)
    alg = TRBDF2(linsolve=KLUFactorization())
    sol = solve(prob, alg; saveat=2.5, parallel)

    ## Now test on a single example 
    x = 0.37
    y = 0.58
    t_idx = 5 # t = sol.t[t_idx]
    V = jump_and_march(prob, (x, y))
    @test DT.is_inside(DT.point_position_relative_to_triangle(get_point(prob, V...)..., (x, y)))
    val = eval_interpolant(sol, x, y, t_idx, V)
    @test eval_interpolant(sol, x, y, t_idx, V) == eval_interpolant(sol, x, y, sol.t[t_idx], V)
    @inferred eval_interpolant(sol, x, y, t_idx, V)
    @inferred eval_interpolant(sol, x, y, sol.t[t_idx], V)
    SHOW_WARNTYPE && @code_warntype eval_interpolant(sol, x, y, t_idx, V)
    SHOW_WARNTYPE && @code_warntype eval_interpolant(sol, x, y, sol.t[t_idx], V)
    SHOW_WARNTYPE && @code_warntype eval_interpolant!(zeros(3), prob, x, y, V, sol.u[5])

    ## Make sure the interpolant is actually an interpolant 
    for j in each_point_index(prob)
        V = jump_and_march(prob, get_point(prob, j))
        @test DT.is_on(DT.point_position_relative_to_triangle(get_point(points, V...)..., get_point(prob, j)))
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
    last_triangle = first(FVM.get_elements(prob)) # initiate 
    for j in 1:ny
        for i in 1:nx
            # global last_triangle
            V_mat[i, j] = jump_and_march(prob, (grid_x[i], grid_y[j]); try_points=last_triangle)
            last_triangle = V_mat[i, j]
            @test !DT.is_outside(DT.point_position_relative_to_triangle(get_point(points, V_mat[i, j]...)..., (grid_x[i], grid_y[j])))
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
    if parallel
        fig = Figure(resolution=(2744.0f0, 692.0f0))
        for k in 1:4
            ax = Axis3(fig[1, k])
            zlims!(ax, 0, 1), xlims!(ax, -L - 1e-1, L + 1e-1), ylims!(ax, -L - 1e-1, L + 1e-1)
            surface!(ax, grid_x, grid_y, u_vals[:, :, k+1], colormap=:matter)
        end
        @test_reference "../docs/src/figures/surface_plots_travelling_wave.png" fig
    end
end
