function get_control_volume(tri, i)
    is_bnd, bnd_idx = DelaunayTriangulation.is_boundary_node(tri, i)
    cv = NTuple{2,Float64}[]
    if is_bnd
        j = DelaunayTriangulation.get_right_boundary_node(tri, i, bnd_idx)
        k = get_adjacent(tri, i, j)
        p = get_point(tri, i)
        push!(cv, p)
        while !DelaunayTriangulation.is_boundary_index(k)
            q, r = get_point(tri, j, k)
            c = (p .+ q .+ r) ./ 3
            m = (p .+ q) ./ 2
            push!(cv, m, c)
            j = k
            k = get_adjacent(tri, i, j)
            DelaunayTriangulation.is_boundary_index(k) && push!(cv, (p .+ r) ./ 2)
        end
        push!(cv, p)
    else
        S = DelaunayTriangulation.get_surrounding_polygon(tri, i)
        push!(S, S[begin])
        j = S[begin]
        p = get_point(tri, i)
        q = get_point(tri, j)
        push!(cv, (p .+ q) ./ 2)
        for k in S[2:end]
            r = get_point(tri, k)
            push!(cv, (p .+ q .+ r) ./ 3)
            push!(cv, (p .+ r) ./ 2)
            q = r
        end
    end
    return cv
end

function random_point_inside_triangle(p, q, r)
    b = q .- p
    c = r .- p
    outside = true
    y = (NaN, NaN)
    while outside
        a₁, a₂ = rand(2)
        x = a₁ .* b .+ a₂ .* c
        y = p .+ x
        outside = DelaunayTriangulation.is_outside(
            DelaunayTriangulation.point_position_relative_to_triangle(p, q, r, y)
        )
    end
    return y
end

function example_tri()
    θ1 = (collect ∘ LinRange)(0, π / 2, 20)
    θ2 = (collect ∘ LinRange)(π / 2, π, 20)
    θ3 = (collect ∘ LinRange)(π, 3π / 2, 20)
    θ4 = (collect ∘ LinRange)(3π / 2, 2π, 20)
    θ4[end] = 0
    x1 = [cos.(θ1), cos.(θ2), cos.(θ3), cos.(θ4)]
    y1 = [sin.(θ1), sin.(θ2), sin.(θ3), sin.(θ4)]
    x1 = [cos.(θ) for θ in (θ1, θ2, θ3, θ4)]
    y1 = [sin.(θ) for θ in (θ1, θ2, θ3, θ4)]
    x2 = [cos.(reverse(θ)) / 2 for θ in (θ4, θ3, θ2, θ1)]
    y2 = [sin.(reverse(θ)) / 2 for θ in (θ4, θ3, θ2, θ1)]
    boundary_nodes, points = convert_boundary_points_to_indices([x1, x2], [y1, y2])
    tri = triangulate(points; boundary_nodes)
    return tri
end

function example_bc_setup()
    f1 = (x, y, t, u, p) -> x * y + u - p
    f2 = (x, y, t, u, p) -> u + p - t
    f3 = (x, y, t, u, p) -> x
    f4 = (x, y, t, u, p) -> y - x
    f5 = (x, y, t, u, p) -> t + p[1] - p[3]
    f6 = (x, y, t, u, p) -> u + p[2] - p[4]
    f7 = (x, y, t, u, p) -> x * t * u
    f8 = (x, y, t, u, p) -> y * t + u
    p1 = 0.7
    p2 = 0.3
    p3 = 0.5
    p4 = 0.2
    p5 = (0.3, 0.6, -1.0)
    p6 = (0.2, 0.4, 0.5, 0.7)
    p7 = nothing
    p8 = nothing
    functions = (f1, f2, f3, f4, f5, f6, f7, f8)
    types = (Dirichlet, Dudt, Neumann, Neumann, Dirichlet, Constrained, Neumann, Dudt)
    parameters = (p1, p2, p3, p4, p5, p6, p7, p8)
    return functions, types, parameters
end

function example_tri_rect()
    a, b, c, d, nx, ny = 0.0, 2.0, 0.0, 5.0, 12, 19
    tri = triangulate_rectangle(a, b, c, d, nx, ny; single_boundary=false, add_ghost_triangles=true)
    return tri
end

function example_problem(idx=1;
    tri=example_tri_rect(),
    mesh=FVMGeometry(tri),
    initial_condition=rand(DelaunayTriangulation.num_solid_vertices(tri)))
    f1 = (x, y, t, u, p) -> x * y + u - p
    f2 = (x, y, t, u, p) -> u + p - t
    f3 = (x, y, t, u, p) -> x
    f4 = (x, y, t, u, p) -> y - x
    f = (f1, f2, f3, f4)
    conditions = (Dirichlet, Neumann, Dirichlet, Neumann)
    parameters = (0.5, 0.2, 0.3, 0.4)
    BCs = BoundaryConditions(mesh, f, conditions; parameters=parameters)
    internal_dirichlet_nodes = Dict([7 + (i - 1) * 12 for i in 2:18] .=> 1)
    ICs = InternalConditions((x, y, t, u, p) -> x + y + t + u + p, ; dirichlet_nodes=internal_dirichlet_nodes, parameters=0.29)
    flux_function = (x, y, t, α, β, γ, p) -> let u = α[idx] * x + β[idx] * y + γ[idx]
        (-α[idx] * u * p[1] + t, x + t - β[idx] * u * p[2])
    end
    flux_parameters = (-0.5, 1.3)
    source_function = (x, y, t, u, p) -> u + p
    source_parameters = 1.5
    initial_time = 2.0
    final_time = 5.0
    prob = FVMProblem(mesh, BCs, ICs;
        flux_function,
        flux_parameters,
        source_function,
        source_parameters,
        initial_condition,
        initial_time,
        final_time)
    return prob, tri, mesh, BCs, ICs, flux_function, flux_parameters, source_function, source_parameters, initial_condition
end

function example_bc_ic_setup(; nothing_dudt=false)
    if !nothing_dudt
        f1 = (x, y, t, u, p) -> x * y + u - p
        f2 = (x, y, t, u, p) -> u + p - t
        f3 = (x, y, t, u, p) -> x
        f4 = (x, y, t, u, p) -> y - x
    else
        f1 = (x, y, t, u, p) -> x * y - p
        f2 = (x, y, t, u, p) -> p
        f3 = (x, y, t, u, p) -> x
        f4 = (x, y, t, u, p) -> y - x
    end
    p1 = 0.5
    p2 = 0.2
    p3 = (0.3, 0.6, -1.0)
    p4 = (0.2, 0.4, 0.5, 0.7)
    f = (f1, f2, f3, f4)
    t = (Dirichlet, Dudt, Neumann, Constrained)
    p = (p1, p2, p3, p4)
    if !nothing_dudt
        g1 = (x, y, t, u, p) -> x * y * t * u * p
        g2 = (x, y, t, u, p) -> x * y * t * u
        g3 = (x, y, t, u, p) -> x * y * t * u * p[2]
        g4 = (x, y, t, u, p) -> x * y * t * u * p[4]
        g5 = (x, y, t, u, p) -> x * y * t * u * p[1]
    else
        g1 = (x, y, t, u, p) -> x * y * p
        g2 = (x, y, t, u, p) -> x * y
        g3 = (x, y, t, u, p) -> x * y * p[2]
        g4 = (x, y, t, u, p) -> x * y * p[4]
        g5 = (x, y, t, u, p) -> x * y * p[1]
    end
    q1 = 0.5
    q2 = nothing
    q3 = (0.3, 0.6, -1.0)
    q4 = (0.2, 0.4, 0.5, 0.7)
    q5 = (0.2, 0.4, 0.5, 0.7)
    g = (g1, g2, g3, g4, g5)
    q = (q1, q2, q3, q4, q5)
    dirichlet_nodes = Dict(
        ([7 + (i - 1) * 12 for i in 1:10] .=> 1)...,
        ([2 + (i - 1) * 12 for i in 7:15] .=> 5)...,
        2 + 2 * 12 => 3
    )
    dudt_nodes = Dict(
        (38:(38+7) .=> 2)...,
        ([9 + (i - 1) * 12 for i in 2:18] .=> 4)...,
    )
    return f, t, p, g, q, dirichlet_nodes, dudt_nodes
end

function test_bc_conditions!(tri, conds, t,
    dirichlet_nodes, dudt_nodes, constrained_edges, neumann_edges, shift=0)
    for e in keys(get_boundary_edge_map(tri))
        u, v = DelaunayTriangulation.edge_indices(e)
        w = get_adjacent(tri, v, u)
        bc_type = t[-w]
        w -= shift
        if bc_type == Dirichlet
            @test FiniteVolumeMethod.has_dirichlet_nodes(conds)
            @test FiniteVolumeMethod.get_dirichlet_nodes(conds) == conds.dirichlet_nodes
            @test conds.dirichlet_nodes[u] == -w
            @test conds.dirichlet_nodes[v] == -w
            @test FiniteVolumeMethod.is_dirichlet_node(conds, u)
            @test FiniteVolumeMethod.is_dirichlet_node(conds, v)
            @test FiniteVolumeMethod.has_condition(conds, u)
            @test FiniteVolumeMethod.has_condition(conds, v)
            @test FiniteVolumeMethod.get_dirichlet_fidx(conds, u) == -w
            @test FiniteVolumeMethod.get_dirichlet_fidx(conds, v) == -w
            push!(dirichlet_nodes, get_point(tri, u, v)...)
            x, y, tt, u = rand(4)
            @test FiniteVolumeMethod.eval_condition_fnc(conds, -w, x, y, tt, u) ≈ conds.functions[-w](x, y, tt, u)
            @inferred conds.functions[-w](x, y, tt, u)
            @inferred FiniteVolumeMethod.eval_condition_fnc(conds, -w, x, y, tt, u)
        elseif bc_type == Dudt
            @test conds.dudt_nodes[u] == -w
            @test conds.dudt_nodes[v] == -w
            @test FiniteVolumeMethod.is_dudt_node(conds, u)
            @test FiniteVolumeMethod.is_dudt_node(conds, v)
            @test FiniteVolumeMethod.has_condition(conds, u)
            @test FiniteVolumeMethod.has_condition(conds, v)
            @test FiniteVolumeMethod.get_dudt_fidx(conds, u) == -w
            @test FiniteVolumeMethod.get_dudt_fidx(conds, v) == -w
            push!(dudt_nodes, get_point(tri, u, v)...)
            x, y, tt, u = rand(4)
            @test FiniteVolumeMethod.eval_condition_fnc(conds, -w, x, y, tt, u) ≈ conds.functions[-w](x, y, tt, u)
            @inferred conds.functions[-w](x, y, tt, u)
            @inferred FiniteVolumeMethod.eval_condition_fnc(conds, -w, x, y, tt, u)
        elseif bc_type == Neumann
            @test conds.neumann_edges[(u, v)] == -w
            push!(neumann_edges, get_point(tri, u, v)...)
            @test (u, v) ∉ keys(conds.constrained_edges)
            @test FiniteVolumeMethod.is_neumann_edge(conds, u, v)
            @test FiniteVolumeMethod.get_neumann_fidx(conds, u, v) == -w
            @test FiniteVolumeMethod.has_neumann_edges(conds)
            x, y, tt, u = rand(4)
            @test FiniteVolumeMethod.eval_condition_fnc(conds, -w, x, y, tt, u) ≈ conds.functions[-w](x, y, tt, u)
            @inferred conds.functions[-w](x, y, tt, u)
            @inferred FiniteVolumeMethod.eval_condition_fnc(conds, -w, x, y, tt, u)
        else
            @test conds.constrained_edges[(u, v)] == -w
            push!(constrained_edges, get_point(tri, u, v)...)
            @test (u, v) ∉ keys(conds.neumann_edges)
            @test FiniteVolumeMethod.is_constrained_edge(conds, u, v)
            @test FiniteVolumeMethod.get_constrained_fidx(conds, u, v) == -w
            x, y, tt, u = rand(4)
            @test FiniteVolumeMethod.eval_condition_fnc(conds, -w, x, y, tt, u) ≈ conds.functions[-w](x, y, tt, u)
            @inferred conds.functions[-w](x, y, tt, u)
            @inferred FiniteVolumeMethod.eval_condition_fnc(conds, -w, x, y, tt, u)
        end
    end
    return nothing
end

function test_bc_ic_conditions!(tri, conds, t,
    dirichlet_nodes, dudt_nodes, constrained_edges, neumann_edges,
    ics)
    nif = length(conds.functions) - DelaunayTriangulation.num_ghost_vertices(tri)
    test_bc_conditions!(tri, conds, t, dirichlet_nodes,
        dudt_nodes, constrained_edges, neumann_edges, nif)
    for (i, idx) in ics.dirichlet_nodes
        @test FiniteVolumeMethod.has_dirichlet_nodes(conds)
        @test FiniteVolumeMethod.get_dirichlet_nodes(conds) == conds.dirichlet_nodes
        if !DelaunayTriangulation.is_boundary_node(tri, i)[1]
            @test conds.dirichlet_nodes[i] == idx
            push!(dirichlet_nodes, get_point(tri, i))
            @test FiniteVolumeMethod.is_dirichlet_node(conds, i)
            @test FiniteVolumeMethod.has_condition(conds, i)
            @test FiniteVolumeMethod.get_dirichlet_fidx(conds, i) == idx
            x, y, t, u = rand(4)
            @test FiniteVolumeMethod.eval_condition_fnc(conds, idx, x, y, t, u) ≈ conds.functions[idx](x, y, t, u)
            @inferred conds.functions[idx](x, y, t, u)
            @inferred FiniteVolumeMethod.eval_condition_fnc(conds, idx, x, y, t, u)
        end
    end
    for (i, idx) in ics.dudt_nodes
        if !DelaunayTriangulation.is_boundary_node(tri, i)[1]
            @test conds.dudt_nodes[i] == idx
            push!(dudt_nodes, get_point(tri, i))
            @test FiniteVolumeMethod.is_dudt_node(conds, i)
            @test FiniteVolumeMethod.has_condition(conds, i)
            @test FiniteVolumeMethod.get_dudt_fidx(conds, i) == idx
            x, y, t, u = rand(4)
            @test FiniteVolumeMethod.eval_condition_fnc(conds, idx, x, y, t, u) ≈ conds.functions[idx](x, y, t, u)
            @inferred conds.functions[idx](x, y, t, u)
            @inferred FiniteVolumeMethod.eval_condition_fnc(conds, idx, x, y, t, u)
        end
    end
end

function example_diffusion_problem()
    a, b, c, d = 0.0, 2.0, 0.0, 2.0
    nx, ny = 25, 25
    tri = triangulate_rectangle(a, b, c, d, nx, ny, single_boundary=true)
    mesh = FVMGeometry(tri)
    bc = (x, y, t, u, p) -> zero(u)
    BCs = BoundaryConditions(mesh, bc, Dirichlet)
    f = (x, y) -> y ≤ 1.0 ? 50.0 : 0.0
    initial_condition = [f(x, y) for (x, y) in each_point(tri)]
    D = (x, y, t, u, p) -> 1 / 9
    final_time = 0.5
    prob = FVMProblem(mesh, BCs; diffusion_function=D, initial_condition, final_time)
    return prob
end
function example_diffusion_problem_system()
    a, b, c, d = 0.0, 2.0, 0.0, 2.0
    nx, ny = 25, 25
    tri = triangulate_rectangle(a, b, c, d, nx, ny, single_boundary=true)
    mesh = FVMGeometry(tri)
    bc = (x, y, t, u, p) -> zero(u[1]) * zero(u[2])
    BCs = BoundaryConditions(mesh, bc, Dirichlet)
    f = (x, y) -> y ≤ 1.0 ? 50.0 : 0.0
    initial_condition = [f(x, y) for (x, y) in each_point(tri)]
    D = (x, y, t, u, p) -> 1 / 9
    q1 = (x, y, t, α, β, γ, p) -> (-α[1] / 9, -β[1] / 9)
    q2 = (x, y, t, α, β, γ, p) -> (-α[2] / 9, -β[2] / 9)
    final_time = 0.5
    prob1 = FVMProblem(mesh, BCs; flux_function=q1, initial_condition, final_time)
    prob2 = FVMProblem(mesh, BCs; flux_function=q2, initial_condition, final_time)
    return FVMSystem(prob1, prob2), example_diffusion_problem()
end

function example_heat_convection_problem()
    L = 1.0
    k = 237.0
    T₀ = 10.0
    T∞ = 10.0
    α = 80e-6
    q = 10.0
    h = 25.0
    tri = triangulate_rectangle(0, L, 0, L, 200, 200; single_boundary=false)
    mesh = FVMGeometry(tri)
    bot_wall = (x, y, t, T, p) -> -p.α * p.q / p.k
    right_wall = (x, y, t, T, p) -> zero(T)
    top_wall = (x, y, t, T, p) -> -p.α * p.h / p.k * (p.T∞ - T)
    left_wall = (x, y, t, T, p) -> zero(T)
    bc_fncs = (bot_wall, right_wall, top_wall, left_wall) # the order is important 
    types = (Neumann, Neumann, Neumann, Neumann)
    bot_parameters = (α=α, q=q, k=k)
    right_parameters = nothing
    top_parameters = (α=α, h=h, k=k, T∞=T∞)
    left_parameters = nothing
    parameters = (bot_parameters, right_parameters, top_parameters, left_parameters)
    BCs = BoundaryConditions(mesh, bc_fncs, types; parameters)
    flux_function = (x, y, t, α, β, γ, p) -> begin
        ∇u = (α, β)
        return -p.α .* ∇u
    end
    flux_parameters = (α=α,)
    final_time = 2000.0
    f = (x, y) -> T₀
    initial_condition = [f(x, y) for (x, y) in each_point(tri)]
    prob = FVMProblem(mesh, BCs;
        flux_function,
        flux_parameters,
        initial_condition,
        final_time)
    return prob
end

function test_shape_function_coefficients(prob, u)
    for T in each_solid_triangle(prob.mesh.triangulation)
        i, j, k = indices(T)
        s₁, s₂, s₃, s₄, s₅, s₆, s₇, s₈, s₉ = prob.mesh.triangle_props[T].shape_function_coefficients
        ui, uj, uk = u[i], u[j], u[k]
        α = s₁ * ui + s₂ * uj + s₃ * uk
        β = s₄ * ui + s₅ * uj + s₆ * uk
        γ = s₇ * ui + s₈ * uj + s₉ * uk
        @test sum(prob.mesh.triangle_props[T].shape_function_coefficients) ≈ 1.0
        xi, yi = get_point(prob.mesh.triangulation, i)
        xj, yj = get_point(prob.mesh.triangulation, j)
        xk, yk = get_point(prob.mesh.triangulation, k)
        @test α * xi + β * yi + γ ≈ ui atol = 1e-9
        @test α * xj + β * yj + γ ≈ uj atol = 1e-9
        @test α * xk + β * yk + γ ≈ uk atol = 1e-9
        cx, cy = (xi + xj + xk) / 3, (yi + yj + yk) / 3
        @test α * cx + β * cy + γ ≈ (ui + uj + uk) / 3
        a, b, c = FVM.get_shape_function_coefficients(prob.mesh.triangle_props[T], T, u, prob)
        @test a ≈ α atol = 1e-9
        @test b ≈ β atol = 1e-9
        @test c ≈ γ atol = 1e-9
        @inferred FVM.get_shape_function_coefficients(prob.mesh.triangle_props[T], T, u, prob)
    end
end

function test_get_flux(prob, u, t)
    for T in each_solid_triangle(prob.mesh.triangulation)
        p, q, r = get_point(prob.mesh.triangulation, T...)
        c = (p .+ q .+ r) ./ 3
        s₁, s₂, s₃, s₄, s₅, s₆, s₇, s₈, s₉ = prob.mesh.triangle_props[T].shape_function_coefficients
        α = s₁ * u[T[1]] + s₂ * u[T[2]] + s₃ * u[T[3]]
        β = s₄ * u[T[1]] + s₅ * u[T[2]] + s₆ * u[T[3]]
        γ = s₇ * u[T[1]] + s₈ * u[T[2]] + s₉ * u[T[3]]
        _qn = Float64[]
        for (edge_index, (i, j)) in enumerate(DelaunayTriangulation.triangle_edges(T))
            p, q = get_point(prob.mesh.triangulation, i, j)
            m = (p .+ q) ./ 2
            x, y = (c .+ m) ./ 2
            ex, ey = c .- m
            ℓ = norm((ex, ey))
            nx, ny = ey / ℓ, -ex / ℓ
            qx, qy = prob.flux_function(x, y, t, α, β, γ, prob.flux_parameters)
            @inferred prob.flux_function(x, y, t, α, β, γ, prob.flux_parameters)
            @inferred FVM.eval_flux_function(prob, x, y, t, α, β, γ)
            qn = (qx * nx + qy * ny) * ℓ
            @test qn ≈ FVM.get_flux(prob, prob.mesh.triangle_props[T], α, β, γ, t, edge_index) atol = 1e-9
            @inferred FVM.get_flux(prob, prob.mesh.triangle_props[T], α, β, γ, t, edge_index)
            push!(_qn, qn)
        end
        @test _qn ≈ collect(FVM.get_fluxes(prob, prob.mesh.triangle_props[T], α, β, γ, t))
        @inferred FVM.get_fluxes(prob, prob.mesh.triangle_props[T], α, β, γ, t)
    end
end

function test_get_boundary_flux(prob, u, t, is_diff=true)
    tri = prob.mesh.triangulation
    for e in keys(get_boundary_edge_map(tri))
        i, j = e
        p, q = get_point(tri, i, j)
        px, py = p
        qx, qy = q
        k = get_adjacent(tri, e)
        T = (i, j, k)
        if haskey(prob.mesh.triangle_props, (j, k, i))
            T = (j, k, i)
        elseif haskey(prob.mesh.triangle_props, (k, i, j))
            T = (k, i, j)
        end
        s₁, s₂, s₃, s₄, s₅, s₆, s₇, s₈, s₉ = prob.mesh.triangle_props[T].shape_function_coefficients
        α = s₁ * u[T[1]] + s₂ * u[T[2]] + s₃ * u[T[3]]
        β = s₄ * u[T[1]] + s₅ * u[T[2]] + s₆ * u[T[3]]
        γ = s₇ * u[T[1]] + s₈ * u[T[2]] + s₉ * u[T[3]]
        if is_diff
            # First edge 
            x, y = (px + (px + qx) / 2) / 2, (py + (py + qy) / 2) / 2
            nx, ny = (qy - py) / norm(p .- q), -(qx - px) / norm(p .- q)
            _qx, _qy = prob.flux_function(x, y, t, α, β, γ, prob.flux_parameters)
            @inferred prob.flux_function(x, y, t, α, β, γ, prob.flux_parameters)
            @inferred FVM.eval_flux_function(prob, x, y, t, α, β, γ)
            q1 = (_qx * nx + _qy * ny) * norm(p .- (p .+ q) ./ 2)
            @test q1 ≈ FVM._get_boundary_flux(prob, x, y, t, α, β, γ, nx, ny, i, j, α * x + β * y + γ) * norm(p .- (p .+ q) ./ 2) atol = 1e-6
            @inferred FVM._get_boundary_flux(prob, x, y, t, α, β, γ, nx, ny, i, j, α * x + β * y + γ)
            # Second edge 
            x, y = ((px + qx) / 2 + qx) / 2, ((py + qy) / 2 + qy) / 2
            _qx, _qy = prob.flux_function(x, y, t, α, β, γ, prob.flux_parameters)
            @inferred prob.flux_function(x, y, t, α, β, γ, prob.flux_parameters)
            @inferred FVM.eval_flux_function(prob, x, y, t, α, β, γ)
            q2 = (_qx * nx + _qy * ny) * norm((p .+ q) ./ 2 .- q)
            @test q2 ≈ FVM._get_boundary_flux(prob, x, y, t, α, β, γ, nx, ny, i, j, α * x + β * y + γ) * norm((p .+ q) ./ 2 .- q)
            @inferred FVM._get_boundary_flux(prob, x, y, t, α, β, γ, nx, ny, i, j, α * x + β * y + γ)
        else
            # First edge
            x, y = (px + (px + qx) / 2) / 2, (py + (py + qy) / 2) / 2
            nx, ny = (qy - py) / norm(p .- q), -(qx - px) / norm(p .- q)
            idx = -get_adjacent(tri, j, i)
            fnc = prob.conditions.functions[idx]
            q1 = fnc(x, y, t, α * x + β * y + γ) * norm(p .- (p .+ q) ./ 2)
            @inferred FVM.eval_condition_fnc(prob, idx, x, y, t, α * x + β * y + γ)
            @test q1 ≈ FVM._get_boundary_flux(prob, x, y, t, α, β, γ, nx, ny, i, j, α * x + β * y + γ) * norm(p .- (p .+ q) ./ 2)
            @inferred FVM._get_boundary_flux(prob, x, y, t, α, β, γ, nx, ny, i, j, α * x + β * y + γ)
            # Second edge
            x, y = ((px + qx) / 2 + qx) / 2, ((py + qy) / 2 + qy) / 2
            nx, ny = (qy - py) / norm(p .- q), -(qx - px) / norm(p .- q)
            idx = -get_adjacent(tri, j, i)
            fnc = prob.conditions.functions[idx]
            q2 = fnc(x, y, t, α * x + β * y + γ) * norm((p .+ q) ./ 2 .- q)
            @inferred FVM.eval_condition_fnc(prob, idx, x, y, t, α * x + β * y + γ)
            @test q2 ≈ FVM._get_boundary_flux(prob, x, y, t, α, β, γ, nx, ny, i, j, α * x + β * y + γ) * norm((p .+ q) ./ 2 .- q)
            @inferred FVM._get_boundary_flux(prob, x, y, t, α, β, γ, nx, ny, i, j, α * x + β * y + γ)
        end
        # Compare
        _q1, _q2 = FVM.get_boundary_fluxes(prob, α, β, γ, e..., t)
        @test q1 ≈ _q1
        @test q2 ≈ _q2
    end
end

function test_single_triangle(prob, u, t)
    du = zero(u)
    for T in each_solid_triangle(prob.mesh.triangulation)
        fill!(du, 0.0)
        p, q, r = get_point(prob.mesh.triangulation, T...)
        c = (p .+ q .+ r) ./ 3
        s₁, s₂, s₃, s₄, s₅, s₆, s₇, s₈, s₉ = prob.mesh.triangle_props[T].shape_function_coefficients
        α = s₁ * u[T[1]] + s₂ * u[T[2]] + s₃ * u[T[3]]
        β = s₄ * u[T[1]] + s₅ * u[T[2]] + s₆ * u[T[3]]
        γ = s₇ * u[T[1]] + s₈ * u[T[2]] + s₉ * u[T[3]]
        _qn = Float64[]
        for (edge_index, (i, j)) in enumerate(DelaunayTriangulation.triangle_edges(T))
            push!(_qn, FVM.get_flux(prob, prob.mesh.triangle_props[T], α, β, γ, t, edge_index))
        end
        q1, q2, q3 = _qn
        dui = -(q1 - q3)
        duj = -(-q1 + q2)
        duk = -(-q2 + q3)
        FVM.fvm_eqs_single_triangle!(du, u, prob, t, T)
        @test du[T[1]] ≈ dui atol = 1e-9
        @test du[T[2]] ≈ duj atol = 1e-9
        @test du[T[3]] ≈ duk atol = 1e-9
    end
end

function test_source_contribution(prob, u, t)
    du = rand(length(u))
    for i in each_solid_vertex(prob.mesh.triangulation)
        _du = du[i] / prob.mesh.cv_volumes[i]
        if i ∈ keys(prob.conditions.dirichlet_nodes)
            _du = 0.0
        end
        FVM.fvm_eqs_single_source_contribution!(du, u, prob, t, i)
        @test du[i] ≈ _du
        @inferred FVM.get_source_contribution(prob, u, t, i)
    end
end

function _is_on_square(p, L)
    x, y = getxy(p)
    on_bot = y ≈ 0.0
    on_right = x ≈ L
    on_top = y ≈ L
    on_left = x ≈ 0.0
    return any((on_bot, on_right, on_top, on_left))
end
function _both_on_same_edge(p, q, L)
    p_on = _is_on_square(p, L)
    q_on = _is_on_square(q, L)
    !(p_on && q_on) && return false, 0
    px, py = p
    qx, qy = q
    on_bot = py ≈ 0.0 && qy ≈ 0.0
    on_right = px ≈ L && qx ≈ L
    on_top = py ≈ L && qy ≈ L
    on_left = px ≈ 0.0 && qx ≈ 0.0
    if on_bot
        return true, 1
    elseif on_right
        return true, 2
    elseif on_top
        return true, 3
    elseif on_left
        return true, 4
    end
end

function get_dudt_val(prob, u, t, i, is_diff=true)
    i ∈ keys(prob.conditions.dirichlet_nodes) && return 0.0
    mesh = prob.mesh
    tri = mesh.triangulation
    cv = get_control_volume(tri, i)
    int = 0.0
    nedges = length(cv) - 1
    for j in 1:nedges
        p, q = cv[j], cv[j+1]
        px, py = p
        qx, qy = q
        mx, my = (px + qx) / 2, (py + qy) / 2
        T = jump_and_march(tri, (mx, my))
        T = DelaunayTriangulation.contains_triangle(tri, T)[1]
        props = mesh.triangle_props[T]
        L = norm(p .- q)
        nx, ny = (qy - py) / L, (px - qx) / L
        s₁, s₂, s₃, s₄, s₅, s₆, s₇, s₈, s₉ = props.shape_function_coefficients
        α = s₁ * u[T[1]] + s₂ * u[T[2]] + s₃ * u[T[3]]
        β = s₄ * u[T[1]] + s₅ * u[T[2]] + s₆ * u[T[3]]
        γ = s₇ * u[T[1]] + s₈ * u[T[2]] + s₉ * u[T[3]]
        if is_diff
            _q = (-α / 9, -β / 9)
            int += L * (_q[1] * nx + _q[2] * ny)
        elseif !is_diff && !_both_on_same_edge(p, q, 1.0)[1]
            _q = (-80e-6 * α, -80e-6 * β)
            int += L * (_q[1] * nx + _q[2] * ny)
        else
            _, idx = _both_on_same_edge(p, q, 1.0)
            if idx == 1
                _q = -80e-6 * 10.0 / 237.0
            elseif idx == 2
                _q = 0.0
            elseif idx == 3
                _q = -80e-6 * 25.0 / 237.0 * (10.0 - (α * mx + β * my + γ))
            elseif idx == 4
                _q = 0.0
            end
            # _q = prob.conditions.functions[idx](mx, my, t, α * mx + β * my + γ)
            int += L * _q
        end
    end
    dudt = prob.source_function(get_point(tri, i)..., t, u[i], prob.source_parameters) - int / mesh.cv_volumes[i]
    return dudt
end

function test_dudt_val(prob, u, t, is_diff=true)
    dudt = [get_dudt_val(prob, u, t, i, is_diff) for i in each_point_index(prob.mesh.triangulation)]
    dudt_fnc = FVM.fvm_eqs!(zero(dudt), u, (prob=prob, parallel=Val(false)), t)
    @test dudt ≈ dudt_fnc
    dudt_fnc = FVM.fvm_eqs!(zero(dudt), u, FVM.get_multithreading_parameters(prob), t)
    @test dudt ≈ dudt_fnc
end

function test_compute_flux(_prob, steady, system, steady_system)
    local prob
    for prob in (_prob, steady, system, steady_system)
        if prob === steady || prob == steady_system
            prob = prob.problem
        end
        if prob isa SteadyFVMProblem
            u = prob.problem.initial_condition
        else
            u = prob.initial_condition
        end
        tri = prob.mesh.triangulation
        for (i, j) in keys(get_boundary_edge_map(tri))
            k = get_adjacent(tri, i, j)
            p, q, r = get_point(tri, i, j, k)
            _i, _j = i, j
            i, j, k = DelaunayTriangulation.contains_triangle(tri, i, j, k)[1]
            props = prob.mesh.triangle_props[(i, j, k)]
            s = props.shape_function_coefficients
            if FVM.is_system(prob)
                α = ntuple(ℓ -> s[1] * u[ℓ, i] + s[2] * u[ℓ, j] + s[3] * u[ℓ, k], FVM._neqs(prob))
                β = ntuple(ℓ -> s[4] * u[ℓ, i] + s[5] * u[ℓ, j] + s[6] * u[ℓ, k], FVM._neqs(prob))
                γ = ntuple(ℓ -> s[7] * u[ℓ, i] + s[8] * u[ℓ, j] + s[9] * u[ℓ, k], FVM._neqs(prob))
            else
                α = s[1] * u[i] + s[2] * u[j] + s[3] * u[k]
                β = s[4] * u[i] + s[5] * u[j] + s[6] * u[k]
                γ = s[7] * u[i] + s[8] * u[j] + s[9] * u[k]
            end
            qv = FVM.eval_flux_function(prob, ((p .+ q) ./ 2)..., 2.5, α, β, γ)
            ex, ey = (q .- p) ./ norm(p .- q)
            nx, ny = ey, -ex
            @test DelaunayTriangulation.distance_to_polygon((p .+ q) ./ 2 .+ (nx, ny), get_points(tri), get_boundary_nodes(tri)) < 0.0
            @test DelaunayTriangulation.is_right(DelaunayTriangulation.point_position_relative_to_line(p, q, (p .+ q) ./ 2 .+ (nx, ny)))
            _qv = compute_flux(prob, _i, _j, u, 2.5)
            if !FVM.is_system(prob)
                @test _qv ≈ dot(qv, (nx, ny))
            else
                all_qvs = ntuple(ℓ -> dot(qv[ℓ], (nx, ny)), FVM._neqs(prob))
                @test collect(_qv) ≈ collect(all_qvs)
                @test all_qvs[1] ≈ compute_flux(prob.problems[1], _i, _j, @views(u[1, :]), 2.5)
            end
            @inferred compute_flux(prob, _i, _j, u, 2.5)
        end
        for (i, j) in each_solid_edge(tri)
            if (i, j) ∉ keys(get_boundary_edge_map(tri))
                p, q = get_point(tri, i, j)
                k = get_adjacent(tri, j, i)
                r = get_point(tri, k)
                a, b, c = j, i, k
                a, b, c = DelaunayTriangulation.contains_triangle(tri, a, b, c)[1]
                props = prob.mesh.triangle_props[(a, b, c)]
                s = props.shape_function_coefficients
                if FVM.is_system(prob)
                    α = ntuple(ℓ -> s[1] * u[ℓ, a] + s[2] * u[ℓ, b] + s[3] * u[ℓ, c], FVM._neqs(prob))
                    β = ntuple(ℓ -> s[4] * u[ℓ, a] + s[5] * u[ℓ, b] + s[6] * u[ℓ, c], FVM._neqs(prob))
                    γ = ntuple(ℓ -> s[7] * u[ℓ, a] + s[8] * u[ℓ, b] + s[9] * u[ℓ, c], FVM._neqs(prob))
                else
                    α = s[1] * u[a] + s[2] * u[b] + s[3] * u[c]
                    β = s[4] * u[a] + s[5] * u[b] + s[6] * u[c]
                    γ = s[7] * u[a] + s[8] * u[b] + s[9] * u[c]
                end
                qv = FVM.eval_flux_function(prob, ((p .+ q) ./ 2)..., 2.5, α, β, γ)
                ex, ey = (q .- p) ./ norm(p .- q)
                nx, ny = ey, -ex
                @test DelaunayTriangulation.is_right(DelaunayTriangulation.point_position_relative_to_line(p, q, (p .+ q) ./ 2 .+ (nx, ny)))
                _qv = compute_flux(prob, i, j, u, 2.5)
                if !FVM.is_system(prob)
                    @test _qv ≈ dot(qv, (nx, ny))
                else
                    all_qvs = ntuple(ℓ -> dot(qv[ℓ], (nx, ny)), FVM._neqs(prob))
                    @test collect(_qv) ≈ collect(all_qvs)
                    @test all_qvs[1] ≈ compute_flux(prob.problems[1], i, j, @views(u[1, :]), 2.5)
                end
                @inferred compute_flux(prob, i, j, u, 2.5)
            end
        end
    end
end

function test_jacobian_sparsity(prob::FVMProblem)
    A = zeros(DelaunayTriangulation.num_solid_vertices(prob.mesh.triangulation), DelaunayTriangulation.num_solid_vertices(prob.mesh.triangulation))
    for i in each_solid_vertex(prob.mesh.triangulation)
        A[i, i] = 1.0
        for j in get_neighbours(tri, i)
            DelaunayTriangulation.is_boundary_index(j) && continue
            A[i, j] = 1.0
        end
    end
    @test A == FVM.jacobian_sparsity(prob)
end

function test_jacobian_sparsity(prob::FVMSystem{N}) where {N}
    tri = prob.mesh.triangulation
    n = DelaunayTriangulation.num_solid_vertices(tri)
    A = zeros(n * N, n * N)
    idx_map = Vector{Vector{Int}}(undef, n)
    idx_mat = zeros(N, n)
    ctr = 1
    for j in 1:n
        for i in 1:N
            idx_mat[i, j] = ctr
            ctr += 1
        end
    end
    for i in 1:n
        idx_map[i] = idx_mat[:, i]
    end
    for i in each_solid_vertex(tri)
        node1 = idx_map[i]
        for j in (i, get_neighbours(tri, i)...)
            DelaunayTriangulation.is_boundary_index(j) && continue
            node2 = idx_map[j]
            for x in node1
                for y in node2
                    A[x, y] = 1.0
                end
            end
        end
    end
    @test A == FVM.jacobian_sparsity(prob)
end
test_jacobian_sparsity(prob::SteadyFVMProblem) = test_jacobian_sparsity(prob.problem)