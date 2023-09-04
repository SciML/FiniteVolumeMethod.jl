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
    g1 = (x, y, t, u, p) -> x*y*t*u*p 
    g2 = (x, y, t, u, p) -> x*y*t*u*p
    g3 = (x, y, t, u, p) -> x*y*t*u*p
    g4 = (x, y, t, u, p) -> x*y*t*u*p
    g5 = (x, y, t, u, p) -> x*y*t*u*p
    q1 = 0.5
    q2 = nothing 
    q3 = (0.3, 0.6, -1.0)
    q4 = (0.2, 0.4, 0.5, 0.7)
    q5 = (0.2, 0.4, 0.5, 0.7)
    g = (g1, g2, g3, g4, g5)
    q = (q1, q2, q3, q4, q5)
end
function test_bc_conditions!(tri, conds, wrap_around, t, 
    dirichlet_nodes, dudt_nodes, constrained_edges, neumann_edges)
    for e in keys(get_boundary_edge_map(tri))
        u, v = DelaunayTriangulation.edge_indices(e)
        w = get_adjacent(tri, v, u)
        bc_type = t[-w]
        if bc_type == Dirichlet
            @test conds.point_conditions[u] == (Dirichlet, -w)
            @test conds.point_conditions[v] == (Dirichlet, -w)
            push!(dirichlet_nodes, get_point(tri, u, v)...)
        elseif bc_type == Dudt
            u_cond, u_idx = conds.point_conditions[u]
            v_cond, v_idx = conds.point_conditions[v]
            if u_cond == Dirichlet
                @test u_idx == -w - 1 && t[u_idx] == Dirichlet
            else
                @test conds.point_conditions[u] == (Dudt, -w)
                push!(dudt_nodes, get_point(tri, u))
            end
            if v_cond == Dirichlet
                @test v_idx == -w - 1 || v_idx == 5 && t[v_idx] == Dirichlet # == 5 because it's wrapped around back to the start of the curve, which starts at index 5
            else
                @test conds.point_conditions[v] == (Dudt, -w)
                push!(dudt_nodes, get_point(tri, v))
            end
        elseif bc_type == Neumann
            @test conds.edge_conditions[(u, v)] == (true, -w)
            push!(neumann_edges, get_point(tri, u, v)...)
        else
            @test conds.edge_conditions[(u, v)] == (false, -w)
            push!(constrained_edges, get_point(tri, u, v)...)
        end
    end
    return nothing
end


@testset "dual_args" begin
    T, U, P = Float64, Float32, String
    @test FVM.unconstrained_dual_arg_types(T, U, P) == let dU = dualgen(U), dT = dualgen(T)
        Tuple{T,T,T,U,P},   # (x, y, t, u, p)
        Tuple{T,T,T,dU,P},  # (x, y, t, dual u, p)
        Tuple{T,T,dT,U,P},  # (x, y, dual t, u, p)
        Tuple{T,T,dT,dU,P}  # (x, y, dual t, dual u, p)
    end
    @test FVM.constrained_dual_arg_types(T, U, P) == let dU = dualgen(U), dT = dualgen(T), point = NTuple{2,T}, edge = NTuple{2,point}
        Tuple{T,T,T,U,edge,P},      # (x, y, t, u, edge, p)
        Tuple{T,T,T,dU,edge,P},     # (x, y, t, dual u, edge, p)
        Tuple{T,T,dT,U,edge,P},     # (x, y, dual t, u, edge, p)
        Tuple{T,T,dT,dU,edge,P},    # (x, y, dual t, dual u, edge, p)
        Tuple{T,T,T,U,point,P},     # (x, y, t, u, point, p)
        Tuple{T,T,T,dU,point,P},    # (x, y, t, dual u, point, p)
        Tuple{T,T,dT,U,point,P},    # (x, y, dual t, u, point, p)
        Tuple{T,T,dT,dU,point,P}    # (x, y, dual t, dual u, point, p)
    end
    @inferred FVM.unconstrained_dual_arg_types(T, U, P)
    @inferred FVM.constrained_dual_arg_types(T, U, P)
    @test FVM.get_dual_arg_types(T, U, P, Val(true)) == (
        FVM.unconstrained_dual_arg_types(T, U, P)...,
        FVM.constrained_dual_arg_types(T, U, P)...
    )
    @test FVM.get_dual_arg_types(T, U, P, Val(false)) == FVM.unconstrained_dual_arg_types(T, U, P)
    @test FVM.get_dual_ret_types(T, U) == let dU = dualgen(U), dT = dualgen(T)
        (U, dU, dT, dualgen(promote_type(U, T))) # (u, dual u, dual t, dual ut)
    end
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
    test_bc_conditions!(tri, conds, 5, t, dirichlet_nodes, dudt_nodes, constrained_edges, neumann_edges)
    fig, ax, sc = triplot(tri, show_constrained_edges=false)
    scatter!(ax, dirichlet_nodes, color=:red, markersize=18)
    scatter!(ax, dudt_nodes, color=:green, markersize=18)
    linesegments!(ax, constrained_edges, color=:blue, linewidth=6)
    linesegments!(ax, neumann_edges, color=:yellow, linewidth=6)
    @test_reference "test_figures/conditions.png" fig
end

