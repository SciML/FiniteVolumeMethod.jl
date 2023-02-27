using ..FiniteVolumeMethod
include("test_setup.jl")
using Test
using CairoMakie
using OrdinaryDiffEq
using LinearSolve
using StatsBase

## Cell model 
function select_refinement_parameter(x, y, left_r, right_r, target_n; tol=200, max_iters=50)
    f = r -> num_points(generate_mesh(x, y, r; gmsh_path=GMSH_PATH)) - target_n
    iters = 1
    fleft = f(left_r)
    fright = f(right_r)
    while iters ≤ max_iters
        middle_r = (left_r + right_r) / 2
        fmiddle = f(middle_r)
        if abs(fmiddle) < tol
            return middle_r
        end
        iters += 1
        if sign(fmiddle) == sign(fleft)
            left_r = middle_r
            fleft = fmiddle
        else
            right_r = middle_r
            fright = fmiddle
        end
    end
    throw("Failed.")
end
a = c = 0.0
b = d = 500.0
_n = 5
x₁ = LinRange(a, b, _n)
y₁ = LinRange(c, c, _n)
x₂ = LinRange(b, b, _n)
y₂ = LinRange(c, d, _n)
x₃ = LinRange(b, a, _n)
y₃ = LinRange(d, d, _n)
x₄ = LinRange(a, a, _n)
y₄ = LinRange(d, c, _n)
x = reduce(vcat, [x₁, x₂, x₃, x₄])
y = reduce(vcat, [y₁, y₂, y₃, y₄])
xy = [[x[i], y[i]] for i in eachindex(x)]
unique!(xy)
x = [xy[i][1] for i in eachindex(xy)]
y = [xy[i][2] for i in eachindex(xy)]
r = select_refinement_parameter(x, y, 2.0, 1000.0, 10_000)
tri = generate_mesh(x, y, r; gmsh_path=GMSH_PATH)
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

alg = TRBDF2(linsolve=KLUFactorization())
sol_par = solve(prob, alg; parallel=true, saveat=0.05)
sol_ser = solve(prob, alg; parallel=false, saveat=0.05)
@test sol_par.u ≈ sol_ser.u

alg = TRBDF2(linsolve=KLUFactorization(; reuse_symbolic=false))
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