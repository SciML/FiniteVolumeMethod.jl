function TestTri()
    x₁ = LinRange(0, 5, 6)
    y₁ = LinRange(0, 0, 6)
    x₂ = LinRange(5, 5, 6)
    y₂ = LinRange(0, 10, 6)
    x₃ = LinRange(5, 0, 6)
    y₃ = LinRange(10, 10, 6)
    x₄ = LinRange(0, 0, 6)
    y₄ = LinRange(10, 0, 6)
    x = [x₁, x₂, x₃, x₄]
    y = [y₁, y₂, y₃, y₄]
    ref = 4.97
    T, adj, adj2v, DG, pts, BN = generate_mesh(x, y, ref; gmsh_path=GMSH_PATH)
    return T, adj, adj2v, DG, pts, BN
end

function DiffusionEquationOnASquarePlate(r=0.1; iip_flux=true)
    ## Generate the mesh 
    x₁ = LinRange(0, 2, 100)
    y₁ = LinRange(0, 0, 100)
    x₂ = LinRange(2, 2, 100)
    y₂ = LinRange(0, 2, 100)
    x₃ = LinRange(2, 0, 100)
    y₃ = LinRange(2, 2, 100)
    x₄ = LinRange(0, 0, 100)
    y₄ = LinRange(2, 0, 100)
    x = reduce(vcat, [x₁, x₂, x₃, x₄])
    y = reduce(vcat, [y₁, y₂, y₃, y₄])
    xy = [[x[i], y[i]] for i in eachindex(x)]
    unique!(xy)
    x = [xy[i][1] for i in eachindex(xy)]
    y = [xy[i][2] for i in eachindex(xy)]
    T, adj, adj2v, DG, points, BN = generate_mesh(x, y, r; gmsh_path=GMSH_PATH)
    DTx = [points[1, i] for i in axes(points, 2)]
    DTy = [points[2, i] for i in axes(points, 2)]
    ## Define the problem 
    mesh = FVMGeometry(T, adj, adj2v, DG, points, BN)
    boundary_functions = Vector{Function}([(x, y, t, p) -> 0.0])
    type = ["Dirichlet"]
    boundary_conditions = BoundaryConditions(mesh, boundary_functions, type)
    f = (x, y) -> y ≤ 1.0 ? 50.0 : 0.0
    D = (x, y, t, u, p) -> 1 // 9
    R = (x, y, t, u, p) -> 0.0
    u₀ = [f(points[:, i]...) for i in axes(points, 2)]
    final_time = 0.5
    prob = FVMProblem(; mesh, iip_flux, boundary_conditions, diffusion=D, reaction=R, diffusion_parameters=9, initial_condition=u₀, final_time)
    return prob, DTx, DTy
end

function DiffusionOnAWedge(; iip_flux=true)
    α = π / 4
    # Bottom edge 
    r₁ = LinRange(0, 1, 100)
    θ₁ = LinRange(0, 0, 100)
    x₁ = @. r₁ * cos(θ₁)
    y₁ = @. r₁ * sin(θ₁)
    # Arc 
    r₂ = LinRange(1, 1, 100)
    θ₂ = LinRange(0, α, 100)
    x₂ = @. r₂ * cos(θ₂)
    y₂ = @. r₂ * sin(θ₂)
    # Upper edge 
    r₃ = LinRange(1, 0, 100)
    θ₃ = LinRange(α, α, 100)
    x₃ = @. r₃ * cos(θ₃)
    y₃ = @. r₃ * sin(θ₃)
    # Now mesh 
    x = [x₁, x₂, x₃]
    y = [y₁, y₂, y₃]
    T, adj, adj2v, DG, points, BN = generate_mesh(x, y, 0.05; gmsh_path=GMSH_PATH)
    DTx = [points[1, i] for i in axes(points, 2)]
    DTy = [points[2, i] for i in axes(points, 2)]
    # Define the problem 
    mesh = FVMGeometry(T, adj, adj2v, DG, points, BN)
    boundary_functions = [(x, y, p) -> 0.0, (x, y, t, p) -> 0.0, (x, y, p) -> 0.0]
    type = ["Neumann", "Dirichlet", "Neumann"]
    boundary_conditions = BoundaryConditions(mesh, boundary_functions, type)
    f = (x, y) -> 1 - sqrt(x^2 + y^2)
    D = (x, y, t, u, p) -> 1.0
    R = (x, y, t, u, p) -> 0.0
    u₀ = [f(points[:, i]...) for i in axes(points, 2)]
    final_time = 0.1
    prob = FVMProblem(; mesh, iip_flux, boundary_conditions, diffusion=D, reaction=R, initial_condition=u₀, final_time)
    return prob, DTx, DTy, α
end

function ReactionDiffusiondudt(; iip_flux=true)
    ## Generate the mesh 
    r = LinRange(1, 1, 1000)
    θ = LinRange(0, 2π, 1000)
    x = @. r * cos(θ)
    y = @. r * sin(θ)
    T, adj, adj2v, DG, points, BN = generate_mesh(x, y, 0.1; gmsh_path=GMSH_PATH)
    DTx = [points[1, i] for i in axes(points, 2)]
    DTy = [points[2, i] for i in axes(points, 2)]
    ## Define the problem 
    mesh = FVMGeometry(T, adj, adj2v, DG, points, BN)
    boundary_functions = Vector{Function}([(x, y, t, u, p) -> u])
    type = ["dudt"]
    boundary_conditions = BoundaryConditions(mesh, boundary_functions, type)
    f = (x, y) -> sqrt(besseli(0.0, sqrt(2) * sqrt(x^2 + y^2)))
    D = (x, y, t, u, p) -> u
    R = (x, y, t, u, p) -> u * (1 - u)
    u₀ = [f(points[:, i]...) for i in axes(points, 2)]
    final_time = 0.10
    prob = FVMProblem(; mesh, iip_flux, boundary_conditions, diffusion=D, reaction=R, initial_condition=u₀, final_time)
    return prob, DTx, DTy
end

function TravellingWaveProblem(; iip_flux=true)
    ## Define the problem 
    a, b, c, d, Nx, Ny = 0.0, 3.0, 0.0, 40.0, 20, 30
    T, adj, adj2v, DG, points, BN = triangulate_structured(a, b, c, d, Nx, Ny; return_boundary_types=true)
    mesh = FVMGeometry(T, adj, adj2v, DG, points, BN)
    DTx = [points[1, i] for i in axes(points, 2)]
    DTy = [points[2, i] for i in axes(points, 2)]
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
    u₀ = [f(points[:, i]...) for i in axes(points, 2)]
    final_time = 50.0
    diffus = 0.9
    prolif = 0.99
    prob = FVMProblem(; mesh, iip_flux, boundary_conditions, diffusion=D, reaction=R, initial_condition=u₀, final_time, reaction_parameters=prolif, diffusion_parameters=diffus)
    return prob, DTx, DTy, diffus, prolif, Nx, Ny, a, b, c, d
end

function error_measure(soln_exact, soln_approx)
    return 100abs.(soln_exact - soln_approx) / maximum(abs.(soln_exact))
end

function DiffusionEquationOnASquarePlateExact(x, y, t, N=200, M=200)
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
function DiffusionOnAWedgeExact(x, y, t, α=π / 4, M=22, N=24)
    f = (r, θ) -> 1.0 - r
    ## Compute the ζ: ζ[m, n+1] is the mth zero of the (nπ/α)th order Bessel function of the first kind 
    ζ = zeros(M, N + 2)
    for n in 0:(N+1)
        order = n * π / α
        try
            @views ζ[:, n+1] .= approx_besselroots(order, M)
        catch
            @views ζ[:, n+1] .= FastGaussQuadrature.besselroots(order, M)
        end
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
function ReactionDiffusiondudtExact(x, y, t)
    u_exact = zeros(length(x))
    for i in eachindex(x)
        u_exact[i] = exp(t) * sqrt(besseli(0.0, sqrt(2) * sqrt(x[i]^2 + y[i]^2)))
    end
    return u_exact
end

function compute_errors(sol, exact, times; error_fnc=error_measure, summary_stat=median)
    errs = Vector{eltype(sol.prob.u0)}([])
    for t in times
        err = error_fnc(exact(t), sol(t))
        push!(errs, err...)
    end
    summary_stat(errs)
end

function TravellingWaveProblemInvarianceErrors(sol, times, Nx, Ny)
    errs = Vector{Float64}([])
    for t in times
        solns = reshape(sol(t), (Nx, Ny))
        for k in 1:Nx
            for j in 1:Nx
                push!(errs, sum(abs.(solns[k, :] - solns[j, :])))
            end
        end
    end
    return median(errs)
end
function TravellingWaveProblemExact(z, zᶜ, cₘᵢₙ)
    z ≤ zᶜ ? 1 - exp(cₘᵢₙ * (z - zᶜ)) : 0.0
end
function TravellingWaveLongTimeErrors(sol, large_t, col, zᶜ, prolif, diffus, C, D, Nx, Ny)
    time_idx = findfirst(sol.t .≥ large_t)
    times = sol.t[time_idx:end]
    c = sqrt(prolif / (2diffus))
    y_vals = @. C + ((1:Ny) - 1) * (D - C) / (Ny - 1)
    cₘᵢₙ = sqrt(prolif * diffus / 2)
    errs = Vector{Float64}([])
    for (i, t) in pairs(times)
        solns = reshape(sol.u[i], (Nx, Ny))[col, :]
        z = @. y_vals - c * t
        exact = TravellingWaveProblemExact.(z, zᶜ, cₘᵢₙ)
        push!(errs, median(abs.(exact - solns)))
    end
    errs
end

function compute_errors(problem::Function, sol, DTx, DTy)
    error_fnc = (u, û) -> abs.(u .- û) ./ max.(eps(Float64), u)
    if problem == DiffusionEquationOnASquarePlate
        times = [0.1, 0.2, 0.3, 0.4, 0.5]
        err = compute_errors(sol, t -> DiffusionEquationOnASquarePlateExact(DTx, DTy, t), times; error_fnc)
        return err
    elseif problem == DiffusionOnAWedge
        times = [0.02, 0.04, 0.06, 0.08, 0.10]
        err = compute_errors(sol, t -> DiffusionOnAWedgeExact(DTx, DTy, t), times; error_fnc)
        return err
    elseif problem == ReactionDiffusiondudt
        times = [0.02, 0.04, 0.06, 0.08, 0.10]
        err = compute_errors(sol, t -> ReactionDiffusiondudtExact(DTx, DTy, t), times; error_fnc)
        return err
    elseif problem == TravellingWaveProblem
        times = [10.0, 20.0, 30.0, 40.0, 50.0]
        err, err2 = TravellingWaveProblemInvarianceErrors(sol, times, 20, 30), median(TravellingWaveLongTimeErrors(sol, 10.0, 8, 1.6, 0.9, 0.99, 0.0, 40.0, 20, 30))
        return 0.5 * (err + err2)
    else
        throw(ArgumentError("Invalid problem specified."))
    end
end