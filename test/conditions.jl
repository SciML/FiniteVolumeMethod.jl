using ..FiniteVolumeMethod
using Test
using LinearAlgebra
using DelaunayTriangulation
using CairoMakie
using ReferenceTests
const FVM = FiniteVolumeMethod
const DT = DelaunayTriangulation
include("test_functions.jl")

@testset "BoundaryConditions" begin
    tri = example_tri()
    mesh = FVMGeometry(tri)
    f, t, p = example_bc_setup()
    BCs = BoundaryConditions(mesh, f, t; parameters = p)
    conds = FVM.Conditions(mesh, BCs)
    @test BCs.condition_types == t
    dirichlet_nodes = NTuple{2, Float64}[]
    dudt_nodes = NTuple{2, Float64}[]
    constrained_edges = NTuple{2, Float64}[]
    neumann_edges = NTuple{2, Float64}[]
    test_bc_conditions!(
        tri, conds, t, dirichlet_nodes, dudt_nodes, constrained_edges, neumann_edges)
    fig, ax, sc = triplot(tri, show_constrained_edges = false)
    scatter!(ax, dudt_nodes, color = :green, markersize = 18)
    scatter!(ax, dirichlet_nodes, color = :red, markersize = 18)
    linesegments!(ax, constrained_edges, color = :blue, linewidth = 6)
    linesegments!(ax, neumann_edges, color = :yellow, linewidth = 6)
    @test_reference "test_figures/conditions.png" fig
    @test FVM.has_dirichlet_nodes(conds)
end

@testset "BoundaryConditions and InternalConditions" begin
    tri = example_tri_rect()
    mesh = FVMGeometry(tri)
    f, t, p, g, q, dirichlet_nodes, dudt_nodes = example_bc_ic_setup()
    BCs = BoundaryConditions(mesh, f, t; parameters = p)
    @test sprint(show, MIME"text/plain"(), BCs) ==
          "BoundaryConditions with $(length(tri.ghost_vertex_ranges)) boundary conditions with types $(BCs.condition_types)"
    ICs = InternalConditions(g; dirichlet_nodes, dudt_nodes, parameters = q)
    @test sprint(show, MIME"text/plain"(), ICs) ==
          "InternalConditions with $(length(ICs.dirichlet_nodes)) Dirichlet nodes and $(length(ICs.dudt_nodes)) Dudt nodes"
    conds = FVM.Conditions(mesh, BCs, ICs)
    @test sprint(show, MIME"text/plain"(), conds) ==
          "Conditions with\n   $(length(conds.neumann_edges)) Neumann edges\n   $(length(conds.constrained_edges)) Constrained edges\n   $(length(conds.dirichlet_nodes)) Dirichlet nodes\n   $(length(conds.dudt_nodes)) Dudt nodes"
    @test FVM.get_f(conds, 1) == conds.functions[1]
    @test FVM.get_f(conds, 2) == conds.functions[2]
    @test BCs.condition_types == t
    @test ICs.dirichlet_nodes == dirichlet_nodes
    @test ICs.dudt_nodes == dudt_nodes
    dirichlet_nodes = NTuple{2, Float64}[]
    dudt_nodes = NTuple{2, Float64}[]
    constrained_edges = NTuple{2, Float64}[]
    neumann_edges = NTuple{2, Float64}[]
    test_bc_ic_conditions!(
        tri, conds, t, dirichlet_nodes, dudt_nodes, constrained_edges, neumann_edges, ICs)
    fig, ax, sc = triplot(tri, show_constrained_edges = false)
    scatter!(ax, dudt_nodes, color = :green, markersize = 18)
    scatter!(ax, dirichlet_nodes, color = :red, markersize = 18)
    linesegments!(ax, constrained_edges, color = :blue, linewidth = 6)
    linesegments!(ax, neumann_edges, color = :yellow, linewidth = 6)
    @test_reference "test_figures/internal_conditions.png" fig
    @test FVM.has_dirichlet_nodes(conds)
    empty!(conds.dirichlet_nodes)
    @test !FVM.has_dirichlet_nodes(conds)
    @test FVM.has_neumann_edges(conds)
    empty!(conds.neumann_edges)
    @test !FVM.has_neumann_edges(conds)
    @test FVM.get_constrained_edges(conds) == conds.constrained_edges
end

@testset "apply_dudt_conditions!" begin
    tri = example_tri_rect()
    mesh = FVMGeometry(tri)
    f, t, p, g, q, dirichlet_nodes, dudt_nodes = example_bc_ic_setup(; nothing_dudt = true)
    BCs = BoundaryConditions(mesh, f, t; parameters = p)
    ICs = InternalConditions(g; dirichlet_nodes, dudt_nodes, parameters = q)
    conds = FVM.Conditions(mesh, BCs, ICs)
    b = zeros(DelaunayTriangulation.num_points(tri))
    FVM.apply_dudt_conditions!(b, mesh, conds)
    for (i, fidx) in dudt_nodes
        if i ∈ keys(conds.dirichlet_nodes)
            @test b[i] == 0.0
        else
            @test b[i] == conds.functions[fidx](get_point(tri, i)..., nothing, nothing)
        end
    end
end

@testset "Flattening and evaluating heterogeneous tuples" begin
    @testset "flatten_tuples" begin # used for combining problems
        tuple1 = Tuple(rand(5))
        tuple2 = Tuple(rand(3))
        tuple3 = Tuple(rand(10))
        tup = (tuple1..., tuple2..., tuple3...)
        @test FVM.flatten_tuples((tuple1, tuple2, tuple3)) == tup
        @test FVM.flatten_tuples((tuple1,)) == tuple1
    end

    @testset "eval_fnc_in_het_tuple" begin
        f1 = (x, y, t, u) -> x * y
        f2 = (x, y, t, u) -> t * u
        f3 = (x, y, t, u) -> 1.0
        f4 = (x, y, t, u) -> 5x
        f5 = (x, y, t, u) -> sin(x)
        f = (f1, f2, f3, f4, f5)
        x, y, t, u = rand(4)
        for i in 1:5
            @test FVM.eval_fnc_in_het_tuple(f, i, x, y, t, u) == f[i](x, y, t, u)
            @inferred FVM.eval_fnc_in_het_tuple(f, i, x, y, t, u)
        end
    end

    @testset "eval_all_fncs_in_tuple" begin
        f1 = (x, y, t, α, β, γ) -> x * y
        f2 = (x, y, t, α, β, γ) -> t * α * x + t * β * y + γ
        f3 = (x, y, t, α, β, γ) -> 1.0
        f4 = (x, y, t, α, β, γ) -> 5x
        f5 = (x, y, t, α, β, γ) -> sin(x)
        f = (f1, f2, f3, f4, f5)
        x, y, t, α, β, γ = rand(6)
        vals = FVM.eval_all_fncs_in_tuple(f, x, y, t, α, β, γ)
        @test vals == ntuple(i -> f[i](x, y, t, α, β, γ), 5)
        @inferred FVM.eval_all_fncs_in_tuple(f, x, y, t, α, β, γ)

        # functions that return tuples
        f1 = (x, y, t, α, β, γ) -> (x * y, -x * y)
        f2 = (x, y, t, α, β, γ) -> (t * α * x + t * β * y + γ, -x)
        f3 = (x, y, t, α, β, γ) -> (1.0, x)
        f4 = (x, y, t, α, β, γ) -> (5x, -y)
        f5 = (x, y, t, α, β, γ) -> (sin(x), cos(x))
        f = (f1, f2, f3, f4, f5)
        x, y, t, α, β, γ = rand(6)
        vals = FVM.eval_all_fncs_in_tuple(f, x, y, t, α, β, γ)
        @test vals == ntuple(i -> f[i](x, y, t, α, β, γ), 5)
        @inferred FVM.eval_all_fncs_in_tuple(f, x, y, t, α, β, γ)
    end
end
