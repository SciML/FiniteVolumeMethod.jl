using ..FiniteVolumeMethod
using Test
using LinearAlgebra
using DelaunayTriangulation
using DiffEqBase: dualgen
using CairoMakie
using ReferenceTests
const FVM = FiniteVolumeMethod
const DT = DelaunayTriangulation
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

@testset "dual_args" begin
    T, U, P = Float64, Float32, String
    @test FVM.get_dual_arg_types(T, U, P) == let dU = dualgen(U), dT = dualgen(T)
        Tuple{T,T,T,U,P},   # (x, y, t, u, p)
        Tuple{T,T,T,dU,P},  # (x, y, t, dual u, p)
        Tuple{T,T,dT,U,P},  # (x, y, dual t, u, p)
        Tuple{T,T,dT,dU,P}  # (x, y, dual t, dual u, p)
    end
    @inferred FVM.get_dual_arg_types(T, U, P)
    @test FVM.get_dual_ret_types(T, U) == let dU = dualgen(U), dT = dualgen(T)
        (U, dU, dT, dualgen(promote_type(U, T))) # (u, dual u, dual t, dual ut)
    end
    @inferred FVM.get_dual_ret_types(T, U)
end

@testset "BoundaryConditions" begin
    tri = example_tri()
    mesh = FVMGeometry(tri)
    f, t, p = example_bc_setup()
    BCs = BoundaryConditions(mesh, f, t; parameters=p)
    conds = FVM.Conditions(mesh, BCs)
    @test conds.parameters == p
    @test conds.unwrapped_functions == f
    @test BCs.parameters == p
    @test BCs.condition_types == t
    @test BCs.unwrapped_functions == f
    dirichlet_nodes = NTuple{2,Float64}[]
    dudt_nodes = NTuple{2,Float64}[]
    constrained_edges = NTuple{2,Float64}[]
    neumann_edges = NTuple{2,Float64}[]
    test_bc_conditions!(tri, conds, t, dirichlet_nodes, dudt_nodes, constrained_edges, neumann_edges)
    fig, ax, sc = triplot(tri, show_constrained_edges=false)
    scatter!(ax, dudt_nodes, color=:green, markersize=18)
    scatter!(ax, dirichlet_nodes, color=:red, markersize=18)
    linesegments!(ax, constrained_edges, color=:blue, linewidth=6)
    linesegments!(ax, neumann_edges, color=:yellow, linewidth=6)
    @test_reference "test_figures/conditions.png" fig
end

@testset "BoundaryConditions and InternalConditions" begin
    tri = example_tri_rect()
    mesh = FVMGeometry(tri)
    f, t, p, g, q, dirichlet_nodes, dudt_nodes = example_bc_ic_setup()
    BCs = BoundaryConditions(mesh, f, t; parameters=p)
    ICs = InternalConditions(g; dirichlet_nodes, dudt_nodes, parameters=q)
    conds = FVM.Conditions(mesh, BCs, ICs)
    @test conds.parameters == (q..., p...)
    @test conds.unwrapped_functions == (g..., f...)
    @test BCs.parameters == p
    @test BCs.condition_types == t
    @test BCs.unwrapped_functions == f
    @test ICs.dirichlet_nodes == dirichlet_nodes
    @test ICs.dudt_nodes == dudt_nodes
    @test ICs.unwrapped_functions == g
    @test ICs.parameters == q
    dirichlet_nodes = NTuple{2,Float64}[]
    dudt_nodes = NTuple{2,Float64}[]
    constrained_edges = NTuple{2,Float64}[]
    neumann_edges = NTuple{2,Float64}[]
    test_bc_ic_conditions!(tri, conds, t, dirichlet_nodes, dudt_nodes, constrained_edges, neumann_edges, ICs)
    fig, ax, sc = triplot(tri, show_constrained_edges=false)
    scatter!(ax, dudt_nodes, color=:green, markersize=18)
    scatter!(ax, dirichlet_nodes, color=:red, markersize=18)
    linesegments!(ax, constrained_edges, color=:blue, linewidth=6)
    linesegments!(ax, neumann_edges, color=:yellow, linewidth=6)
    @test_reference "test_figures/internal_conditions.png" fig
end