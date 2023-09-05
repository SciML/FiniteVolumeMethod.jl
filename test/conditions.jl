using ..FiniteVolumeMethod
using Test
using LinearAlgebra
using DelaunayTriangulation
using DiffEqBase: dualgen
using CairoMakie
using ReferenceTests
const FVM = FiniteVolumeMethod
const DT = DelaunayTriangulation
include("test_functions.jl")

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
    @test FVM.has_dirichlet_nodes(conds)
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
    @test FVM.has_dirichlet_nodes(conds)
    empty!(conds.dirichlet_nodes)
    @test !FVM.has_dirichlet_nodes(conds)
end