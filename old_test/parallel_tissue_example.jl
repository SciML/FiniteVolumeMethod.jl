using ..FiniteVolumeMethod
include("test_setup.jl")
using Test
using CairoMakie
using OrdinaryDiffEq
using LinearSolve
using StatsBase

## Cell model 
a = c = 0.0
b = d = 500.0
x = [a,b,b,a,a]
y = [c,c,d,d,c]
boundary_nodes, points = convert_boundary_points_to_indices(x, y)
tri = triangulate(points; boundary_nodes)
A = get_total_area(tri)
refine!(tri; max_area=1e-4A)
msh = FVMGeometry(tri)
functions = (x, y, t, u, p) -> p[1] * u * (1 - u)
type = :dudt
params = [[0.5]]
BC = BoundaryConditions(msh, functions, type; params)
Rf = (x, y, t, u, p) -> p[1] * u * (1 - u)
function flux_fnc!(q, x, y, t, α, β, γ, p)
    local D
    D = p[1]
    u = α * x + β * y + γ
    q[1] = -D * u * α
    q[2] = -D * u * β
    return nothing
end
flux_parameters = [150.0]
reaction_parameters = [0.5]
initial_condition = zeros(num_points(tri))
initial_condition[get_boundary_nodes(tri)] .= 0.25
prob = FVMProblem(msh, BC;
    flux_function=flux_fnc!,
    reaction_function=Rf,
    flux_parameters=flux_parameters,
    reaction_parameters=reaction_parameters,
    initial_time=4.0,
    final_time=48.0,
    initial_condition=initial_condition)

alg = TRBDF2(linsolve=KLUFactorization(), autodiff = VERSION ≥ v"1.8.5")
sol_par = solve(prob, alg; parallel=true, saveat=0.05)
sol_ser = solve(prob, alg; parallel=false, saveat=0.05)
@test sol_par.u ≈ sol_ser.u

alg = TRBDF2(linsolve=KLUFactorization(; reuse_symbolic=false), autodiff = VERSION ≥ v"1.8.5")
sol_par = solve(prob, alg; parallel=true, saveat=0.05)
sol_ser = solve(prob, alg; parallel=false, saveat=0.05)
@test sol_par.u ≈ sol_ser.u

alg = TRBDF2(linsolve=KLUFactorization(), autodiff=false)
sol_par = solve(prob, alg; parallel=true, saveat=0.05)
sol_ser = solve(prob, alg; parallel=false, saveat=0.05)
@test sol_par.u ≈ sol_ser.u

alg = TRBDF2(linsolve=KLUFactorization(; reuse_symbolic=false), autodiff=false)
sol_par = solve(prob, alg; parallel=true, saveat=0.05)
sol_ser = solve(prob, alg; parallel=false, saveat=0.05)
@test sol_par.u ≈ sol_ser.u