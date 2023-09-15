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
    BCs = BoundaryConditions(mesh, f, t; parameters=p)
    conds = FVM.Conditions(mesh, BCs)
    @test BCs.condition_types == t
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
    @test FVM.has_dirichlet_nodes(conds)
end

@testset "BoundaryConditions and InternalConditions" begin
    tri = example_tri_rect()
    mesh = FVMGeometry(tri)
    f, t, p, g, q, dirichlet_nodes, dudt_nodes = example_bc_ic_setup()
    BCs = BoundaryConditions(mesh, f, t; parameters=p)
    @test sprint(show, MIME"text/plain"(), BCs) == "BoundaryConditions with $(length(tri.boundary_index_ranges)) boundary conditions with types $(BCs.condition_types)"
    ICs = InternalConditions(g; dirichlet_nodes, dudt_nodes, parameters=q)
    @test sprint(show, MIME"text/plain"(), ICs) == "InternalConditions with $(length(ICs.dirichlet_nodes)) Dirichlet nodes and $(length(ICs.dudt_nodes)) Dudt nodes"
    conds = FVM.Conditions(mesh, BCs, ICs)
    @test sprint(show, MIME"text/plain"(), conds) == "Conditions with\n   $(length(conds.neumann_edges)) Neumann edges\n   $(length(conds.constrained_edges)) Constrained edges\n   $(length(conds.dirichlet_nodes)) Dirichlet nodes\n   $(length(conds.dudt_nodes)) Dudt nodes"
    @test FVM.get_f(conds, 1) == conds.functions[1]
    @test FVM.get_f(conds, 2) == conds.functions[2]
    @test BCs.condition_types == t
    @test ICs.dirichlet_nodes == dirichlet_nodes
    @test ICs.dudt_nodes == dudt_nodes
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
    f, t, p, g, q, dirichlet_nodes, dudt_nodes = example_bc_ic_setup(; nothing_dudt=true)
    BCs = BoundaryConditions(mesh, f, t; parameters=p)
    ICs = InternalConditions(g; dirichlet_nodes, dudt_nodes, parameters=q)
    conds = FVM.Conditions(mesh, BCs, ICs)
    b = zeros(num_points(tri))
    FVM.apply_dudt_conditions!(b, mesh, conds)
    for (i, fidx) in dudt_nodes
        if i ∈ keys(conds.dirichlet_nodes)
            @test b[i] == 0.0
        else
            @test b[i] == conds.functions[fidx](get_point(tri, i)..., nothing, nothing)
        end
    end
end