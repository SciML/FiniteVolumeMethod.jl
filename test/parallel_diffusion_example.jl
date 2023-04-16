using ..FiniteVolumeMethod
include("test_setup.jl")
using Test
using CairoMakie
using OrdinaryDiffEq
using LinearSolve
using PreallocationTools
using StatsBase
using StableRNGs

a, b, c, d = 0.0, 2.0, 0.0, 2.0
p1 = (a, c)
p2 = (b, c)
p3 = (b, d)
p4 = (a, d)
points = [p1, p2, p3, p4]
rng = StableRNG(19191919)
boundary_nodes = [1, 2, 3, 4, 1]
tri = triangulate(points; boundary_nodes, rng)
A = get_total_area(tri)
refine!(tri; max_area=1e-4A, rng)
mesh = FVMGeometry(tri)
bc = ((x, y, t, u::T, p) where {T}) -> zero(T)
type = :Dirichlet
BCs = BoundaryConditions(mesh, bc, type)
f = (x, y) -> y ≤ 1.0 ? 50.0 : 0.0
D = (x, y, t, u, p) -> 1.0 / 9.0
R = ((x, y, t, u::T, p) where {T}) -> zero(T)
points = get_points(tri)
u₀ = f.(first.(points), last.(points))
iip_flux = true
final_time = 48.0
prob = FVMProblem(mesh, BCs; iip_flux=true,
    diffusion_function=D, reaction_function=R,
    initial_condition=u₀, final_time)

u0 = FVM.get_initial_condition(prob)
du_copies,
flux_caches,
shape_coeffs,
dudt_nodes,
interior_or_neumann_nodes,
boundary_elements,
interior_elements,
elements,
dirichlet_nodes,
chunked_boundary_elements,
chunked_interior_elements,
chunked_elements = FVM.prepare_vectors_for_multithreading(u0, prob, Float64; chunk_size=12)
p = (
    prob,
    du_copies,
    flux_caches,
    shape_coeffs,
    dudt_nodes,
    interior_or_neumann_nodes,
    boundary_elements,
    interior_elements,
    elements,
    dirichlet_nodes,
    chunked_boundary_elements,
    chunked_interior_elements,
    chunked_elements
)
prob,#1
du_copies,#2
flux_caches,#3
shape_coeffs,#4
dudt_nodes,#5
interior_or_neumann_nodes,#6
boundary_elements,#7
interior_elements,#8
elements,#9
dirichlet_nodes,#10
chunked_boundary_elements,#11
chunked_interior_elements,#12
chunked_elements = p

flux_cache = PreallocationTools.DiffCache(zeros(Float64, 2), 12)
shape_coeff = PreallocationTools.DiffCache(zeros(Float64, 3), 12)

u = deepcopy(prob.initial_condition)
du_serial = zero(u)
fill!(du_serial, 0.0)

du_parallel = get_tmp(du_copies, u)
flat_du_parallel = zero(u)
fill!(du_parallel, 0.0)

FVM.par_fvm_eqs_interior_element!(du_parallel, u, 0.0, prob, interior_elements, chunked_interior_elements, flux_caches, shape_coeffs)
tmp_flux_cache = get_tmp(flux_cache, u)
tmp_shape_coeffs = get_tmp(shape_coeff, u)
FVM.fvm_eqs_interior_element!(du_serial, u, 0.0, prob, tmp_shape_coeffs, tmp_flux_cache)
@test sum(du_parallel; dims=2) ≈ du_serial

FVM.par_fvm_eqs_boundary_element!(du_parallel, u, 0.0, prob, boundary_elements, chunked_boundary_elements, flux_caches, shape_coeffs)
FVM.fvm_eqs_boundary_element!(du_serial, u, 0.0, prob, tmp_shape_coeffs, tmp_flux_cache)
@test sum(du_parallel; dims=2) ≈ du_serial

for _du in eachcol(du_parallel)
    flat_du_parallel .+= _du
end

@test flat_du_parallel ≈ du_serial

FVM.par_fvm_eqs_source_contribution!(flat_du_parallel, u, 0.0, prob, interior_or_neumann_nodes)
FVM.fvm_eqs_source_contribution!(du_serial, u, 0.0, prob)
@test flat_du_parallel ≈ du_serial

FVM.par_update_dudt_node!(flat_du_parallel, u, 0.0, prob, dudt_nodes)
FVM.update_dudt_nodes!(du_serial, u, 0.0, prob)
@test flat_du_parallel ≈ du_serial

sol_par = solve(prob, TRBDF2(linsolve=KLUFactorization(), autodiff=true); parallel=true, saveat=0.05)
sol_ser = solve(prob, TRBDF2(linsolve=KLUFactorization(), autodiff=true); parallel=false, saveat=0.05)
@test sol_par.u ≈ sol_ser.u

prob = FVMProblem(mesh, BCs; iip_flux=false,
    diffusion_function=D, reaction_function=R,
    initial_condition=u₀, final_time
)
sol_par = solve(prob, TRBDF2(linsolve=KLUFactorization(), autodiff=true); parallel=true, saveat=0.05)
sol_ser = solve(prob, TRBDF2(linsolve=KLUFactorization(), autodiff=true); parallel=false, saveat=0.05)
@test sol_par.u ≈ sol_ser.u