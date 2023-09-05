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

function example_problem()
    tri = example_tri_rect()
    mesh = FVMGeometry(tri)
    f1 = (x, y, t, u, p) -> x * y + u - p
    f2 = (x, y, t, u, p) -> u + p - t
    f3 = (x, y, t, u, p) -> x
    f4 = (x, y, t, u, p) -> y - x
    f = (f1, f2, f3, f4)
    conditions = (Dirichlet, Neumann, Dirichlet, Neumann)
    parameters = (0.5, 0.2, 0.3, 0.4)
    BCs = BoundaryConditions(mesh, f, conditions; parameters=parameters)
    internal_dirichlet_nodes = Dict([7 + (i - 1) * 12 for i in 2:18] .=> 1)
    ICs = InternalConditions(((x, y, t, u, p) -> zero(u),); dirichlet_nodes=internal_dirichlet_nodes)
    flux_function = (x, y, t, α, β, γ, p) -> let u = α * x + β * y + γ
        (-α * u * p[1] + t, x + t - β * u * p[2])
    end
    flux_parameters = (-0.5, 1.3)
    source_function = (x, y, t, u, p) -> u + p
    source_parameters = 1.5
    initial_condition = rand(num_points(tri))
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

function example_bc_ic_setup()
    f1 = (x, y, t, u, p) -> x * y + u - p
    f2 = (x, y, t, u, p) -> u + p - t
    f3 = (x, y, t, u, p) -> x
    f4 = (x, y, t, u, p) -> y - x
    p1 = 0.5
    p2 = 0.2
    p3 = (0.3, 0.6, -1.0)
    p4 = (0.2, 0.4, 0.5, 0.7)
    f = (f1, f2, f3, f4)
    t = (Dirichlet, Dudt, Neumann, Constrained)
    p = (p1, p2, p3, p4)
    g1 = (x, y, t, u, p) -> x * y * t * u * p
    g2 = (x, y, t, u, p) -> x * y * t * u * p
    g3 = (x, y, t, u, p) -> x * y * t * u * p
    g4 = (x, y, t, u, p) -> x * y * t * u * p
    g5 = (x, y, t, u, p) -> x * y * t * u * p
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
            @test conds.dirichlet_nodes[u] == -w
            @test conds.dirichlet_nodes[v] == -w
            push!(dirichlet_nodes, get_point(tri, u, v)...)
        elseif bc_type == Dudt
            @test conds.dudt_nodes[u] == -w
            @test conds.dudt_nodes[v] == -w
            push!(dudt_nodes, get_point(tri, u, v)...)
        elseif bc_type == Neumann
            @test conds.neumann_edges[(u, v)] == -w
            push!(neumann_edges, get_point(tri, u, v)...)
            @test (u, v) ∉ keys(conds.constrained_edges)
        else
            @test conds.constrained_edges[(u, v)] == -w
            push!(constrained_edges, get_point(tri, u, v)...)
            @test (u, v) ∉ keys(conds.neumann_edges)
        end
    end
    return nothing
end

function test_bc_ic_conditions!(tri, conds, t,
    dirichlet_nodes, dudt_nodes, constrained_edges, neumann_edges,
    ics)
    nif = length(conds.parameters) - DelaunayTriangulation.num_ghost_vertices(tri)
    test_bc_conditions!(tri, conds, t, dirichlet_nodes,
        dudt_nodes, constrained_edges, neumann_edges, nif)
    for (i, idx) in ics.dirichlet_nodes
        if !DelaunayTriangulation.is_boundary_node(tri, i)[1]
            @test conds.dirichlet_nodes[i] == idx
            push!(dirichlet_nodes, get_point(tri, i))
        end
    end
    for (i, idx) in ics.dudt_nodes
        if !DelaunayTriangulation.is_boundary_node(tri, i)[1]
            @test conds.dudt_nodes[i] == idx
            push!(dudt_nodes, get_point(tri, i))
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
            qn = (qx * nx + qy * ny) * ℓ
            @test qn ≈ FVM.get_flux(prob, prob.mesh.triangle_props[T], α, β, γ, t, i, j, edge_index) atol = 1e-9
            push!(_qn, qn)
        end
        @test _qn ≈ collect(FVM.get_fluxes(prob, prob.mesh.triangle_props[T], α, β, γ, T..., t)) atol = 1e-9
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
            push!(_qn, FVM.get_flux(prob, prob.mesh.triangle_props[T], α, β, γ, t, i, j, edge_index))
        end
        q1, q2, q3 = _qn 
        dui = -(q1 - q3) 
        duj = -(-q1 + q2)
        duk = -(-q2 + q3)
        FVM.fvm_eqs_single_triangle!(du,u,prob,t,T)
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
    end
end

function get_dudt_val(prob, u, t, i)
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
        q = (-α / 9, -β / 9)
        int += L * (q[1] * nx + q[2] * ny)
    end
    dudt = prob.source_function(get_point(tri, i)..., t, u[i], prob.source_parameters) - int / mesh.cv_volumes[i]
    return dudt
end

function test_dudt_val(prob, u, t)
    dudt = [get_dudt_val(prob, u, t, i) for i in each_point_index(prob.mesh.triangulation)]
    dudt_fnc = FVM.fvm_eqs!(zero(dudt), u, (prob=prob, parallel=Val(false)), t)
    @test dudt ≈ dudt_fnc
end