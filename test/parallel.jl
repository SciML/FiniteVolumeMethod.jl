### Test all the parallel equations

# Diffusion equation on a square plate problem 
a, b, c, d = 0.0, 2.0, 0.0, 2.0
n = 500
x₁ = LinRange(a, b, n)
x₂ = LinRange(b, b, n)
x₃ = LinRange(b, a, n)
x₄ = LinRange(a, a, n)
y₁ = LinRange(c, c, n)
y₂ = LinRange(c, d, n)
y₃ = LinRange(d, d, n)
y₄ = LinRange(d, c, n)
x = reduce(vcat, [x₁, x₂, x₃, x₄])
y = reduce(vcat, [y₁, y₂, y₃, y₄])
xy = [[x[i], y[i]] for i in eachindex(x)]
unique!(xy)
x = getx.(xy)
y = gety.(xy)
r = 0.017
T, adj, adj2v, DG, points, BN = generate_mesh(x, y, r; gmsh_path=GMSH_PATH)
mesh = FVMGeometry(T, adj, adj2v, DG, points, BN)
bc = ((x, y, t, u::T, p) where {T}) -> zero(T)
type = :Dirichlet
BCs = BoundaryConditions(mesh, bc, type, BN)
f = (x, y) -> y ≤ 1.0 ? 50.0 : 0.0
D = (x, y, t, u, p) -> 1.0 / 9.0
R = ((x, y, t, u::T, p) where {T}) -> zero(T)
u₀ = @views f.(points[1, :], points[2, :])
iip_flux = true
final_time = 48.0
prob = FVMProblem(mesh, BCs; iip_flux=true,
    diffusion_function=D, reaction_function=R,
    initial_condition=u₀, final_time)

u0 = FiniteVolumeMethod.get_initial_condition(prob)
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
chunked_elements = FiniteVolumeMethod.prepare_vectors_for_multithreading(u0, prob, Float64; chunk_size=12)
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

FiniteVolumeMethod.par_fvm_eqs_interior_element!(du_parallel, u, 0.0, prob, interior_elements, chunked_interior_elements, flux_caches, shape_coeffs)
tmp_flux_cache = get_tmp(flux_cache, u)
tmp_shape_coeffs = get_tmp(shape_coeff, u)
FiniteVolumeMethod.fvm_eqs_interior_element!(du_serial, u, 0.0, prob, tmp_shape_coeffs, tmp_flux_cache)
@test sum(du_parallel; dims=2) ≈ du_serial

FiniteVolumeMethod.par_fvm_eqs_boundary_element!(du_parallel, u, 0.0, prob, boundary_elements, chunked_boundary_elements, flux_caches, shape_coeffs)
FiniteVolumeMethod.fvm_eqs_boundary_element!(du_serial, u, 0.0, prob, tmp_shape_coeffs, tmp_flux_cache)
@test sum(du_parallel; dims=2) ≈ du_serial

for _du in eachcol(du_parallel)
    flat_du_parallel .+= _du
end

@test flat_du_parallel ≈ du_serial

FiniteVolumeMethod.par_fvm_eqs_source_contribution!(flat_du_parallel, u, 0.0, prob, interior_or_neumann_nodes)
FiniteVolumeMethod.fvm_eqs_source_contribution!(du_serial, u, 0.0, prob)
@test flat_du_parallel ≈ du_serial

FiniteVolumeMethod.par_update_dudt_node!(flat_du_parallel, u, 0.0, prob, dudt_nodes)
FiniteVolumeMethod.update_dudt_nodes!(du_serial, u, 0.0, prob)
@test flat_du_parallel ≈ du_serial

sol_par = solve(prob, TRBDF2(linsolve=KLUFactorization(), autodiff=true); parallel=true, saveat=0.05)
sol_ser = solve(prob, TRBDF2(linsolve=KLUFactorization(), autodiff=true); parallel=false, saveat=0.05)
@test sol_par.u ≈ sol_ser.u

using BenchmarkTools

#sol_par = @benchmark solve($prob, $TRBDF2(linsolve=KLUFactorization(), autodiff=$false); parallel=$true,specialization=$SciMLBase.FullSpecialize, saveat=$0.05)
#sol_ser = @benchmark solve($prob, $TRBDF2(linsolve=KLUFactorization(), autodiff=$false); parallel=$false,specialization=$SciMLBase.FullSpecialize, saveat=$0.05)


prob = FVMProblem(mesh, BCs; iip_flux=false,
    diffusion_function=D, reaction_function=R,
    initial_condition=u₀, final_time
)
sol_par = solve(prob, TRBDF2(linsolve=KLUFactorization(), autodiff=true); parallel=true, saveat=0.05)
sol_ser = solve(prob, TRBDF2(linsolve=KLUFactorization(), autodiff=true); parallel=false, saveat=0.05)
@test sol_par.u ≈ sol_ser.u

#sol_par_flux_oop = @benchmark solve($prob, $TRBDF2(linsolve=KLUFactorization(), autodiff=$true); parallel=$true,specialization=$SciMLBase.FullSpecialize, saveat=$0.05)
#sol_ser_flux_oop = @benchmark solve($prob, $TRBDF2(linsolve=KLUFactorization(), autodiff=$true); parallel=$false,specialization=$SciMLBase.FullSpecialize, saveat=$0.05)

## Diffusion equation on a wedge 
n = 500
α = π / 4
r₁ = LinRange(0, 1, n)
θ₁ = LinRange(0, 0, n)
x₁ = @. r₁ * cos(θ₁)
y₁ = @. r₁ * sin(θ₁)
r₂ = LinRange(1, 1, n)
θ₂ = LinRange(0, α, n)
x₂ = @. r₂ * cos(θ₂)
y₂ = @. r₂ * sin(θ₂)
r₃ = LinRange(1, 0, n)
θ₃ = LinRange(α, α, n)
x₃ = @. r₃ * cos(θ₃)
y₃ = @. r₃ * sin(θ₃)
x = [x₁, x₂, x₃]
y = [y₁, y₂, y₃]
r = 0.01
T, adj, adj2v, DG, points, BN = generate_mesh(x, y, r; gmsh_path=GMSH_PATH)
mesh = FVMGeometry(T, adj, adj2v, DG, points, BN)
lower_bc = ((x, y, t, u::T, p) where {T}) -> zero(T)
arc_bc = ((x, y, t, u::T, p) where {T}) -> zero(T)
upper_bc = ((x, y, t, u::T, p) where {T}) -> zero(T)
types = (:N, :D, :N)
boundary_functions = (lower_bc, arc_bc, upper_bc)
BCs = BoundaryConditions(mesh, boundary_functions, types, BN)
f = (x, y) -> 1 - sqrt(x^2 + y^2)
D = ((x, y, t, u::T, p) where {T}) -> one(T)
u₀ = f.(points[1, :], points[2, :])
final_time = 20.0
prob = FVMProblem(mesh, BCs; iip_flux=true, diffusion_function=D, initial_condition=u₀, final_time)
sol_par = solve(prob, TRBDF2(linsolve=KLUFactorization(), autodiff=true); parallel=true, saveat=0.05)
sol_ser = solve(prob, TRBDF2(linsolve=KLUFactorization(), autodiff=true); parallel=false, saveat=0.05)
@test sol_par.u ≈ sol_ser.u

#sol_par_flux = @benchmark solve($prob, $TRBDF2(linsolve=KLUFactorization(), autodiff=$true); parallel=$true, saveat=$0.05)
#sol_serf = @benchmark solve($prob, $TRBDF2(linsolve=KLUFactorization(), autodiff=$true); parallel=$false, saveat=$0.05)

## Reaction-diffusion equation 
n = 500
r = LinRange(1, 1, 1000)
θ = LinRange(0, 2π, 1000)
x = @. r * cos(θ)
y = @. r * sin(θ)
r = 0.02
T, adj, adj2v, DG, points, BN = generate_mesh(x, y, r; gmsh_path=GMSH_PATH)
mesh = FVMGeometry(T, adj, adj2v, DG, points, BN)
bc = (x, y, t, u, p) -> u
types = :dudt
BCs = BoundaryConditions(mesh, bc, types, BN)
f = (x, y) -> sqrt(besseli(0.0, sqrt(2) * sqrt(x^2 + y^2)))
D = (x, y, t, u, p) -> u
R = (x, y, t, u, p) -> u * (1 - u)
u₀ = [f(points[:, i]...) for i in axes(points, 2)]
final_time = 0.5
prob = FVMProblem(mesh, BCs; iip_flux=false, diffusion_function=D, reaction_function=R, initial_condition=u₀, final_time)
alg = TRBDF2(linsolve=KLUFactorization(; reuse_symbolic=false))
sol_par = solve(prob, alg; parallel=true, saveat=0.05)
sol_ser = solve(prob, alg; parallel=false, saveat=0.05)
@test sol_par.u ≈ sol_ser.u

prob = FVMProblem(mesh, BCs; iip_flux=true, diffusion_function=D, reaction_function=R, initial_condition=u₀, final_time)
alg = TRBDF2(linsolve=KLUFactorization(; reuse_symbolic=false))
sol_par = solve(prob, alg; parallel=true, saveat=0.05)
sol_ser = solve(prob, alg; parallel=false, saveat=0.05)
@test sol_par.u ≈ sol_ser.u

#=
sol_par_flux = @benchmark solve($prob, $TRBDF2(linsolve=KLUFactorization(), autodiff=$true); parallel=$true, saveat=$0.05)
sol_serf = @benchmark solve($prob, $TRBDF2(linsolve=KLUFactorization(), autodiff=$true); parallel=$false, saveat=$0.05)

sol_par_flux2 = @benchmark solve($prob, $TRBDF2(linsolve=KLUFactorization(;reuse_symbolic=false), autodiff=$true); parallel=$true, saveat=$0.05)
sol_serf2 = @benchmark solve($prob, $TRBDF2(linsolve=KLUFactorization(;reuse_symbolic=false), autodiff=$true); parallel=$false, saveat=$0.05)
=#

## Cell model 
function select_refinement_parameter(x, y, left_r, right_r, target_n; tol=200, max_iters=50)
    f = r -> size(generate_mesh(x, y, r; gmsh_path=GMSH_PATH)[5], 2) - target_n
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
_n = 500
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
T, adj, adj2v, DG, pts, BN = generate_mesh(x, y, r; gmsh_path=GMSH_PATH)
msh = FVMGeometry(T, adj, adj2v, DG, pts, BN)
functions = (x, y, t, u, p) -> p[1] * u * (1 - u)
type = :dudt
params = [[0.5]]
BC = BoundaryConditions(msh, functions, type, BN; params)
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
initial_condition = zeros(num_points(pts))
initial_condition[BC.boundary_node_vector[1]] .= 0.25
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

#par_t = @benchmark solve($prob, $alg; parallel=$true, saveat=$4.0)
#ser_t = @benchmark solve($prob, $alg; parallel=$false, saveat=$4.0)

alg = TRBDF2(linsolve=KLUFactorization(; reuse_symbolic=false))
sol_par = solve(prob, alg; parallel=true, saveat=0.05)
sol_ser = solve(prob, alg; parallel=false, saveat=0.05)
@test sol_par.u ≈ sol_ser.u

#par_t = @benchmark solve($prob, $alg; parallel=$true, saveat=$4.0)
#ser_t = @benchmark solve($prob, $alg; parallel=$false, saveat=$4.0)

alg = TRBDF2(linsolve=KLUFactorization(), autodiff=false)
sol_par = solve(prob, alg; parallel=true, saveat=0.05)
sol_ser = solve(prob, alg; parallel=false, saveat=0.05)
@test sol_par.u ≈ sol_ser.u

#par_t = @benchmark solve($prob, $alg; parallel=$true, saveat=$4.0)
#ser_t = @benchmark solve($prob, $alg; parallel=$false, saveat=$4.0)

alg = TRBDF2(linsolve=KLUFactorization(;reuse_symbolic=false), autodiff=false)
sol_par = solve(prob, alg; parallel=true, saveat=0.05)
sol_ser = solve(prob, alg; parallel=false, saveat=0.05)
@test sol_par.u ≈ sol_ser.u

#par_t = @benchmark solve($prob, $alg; parallel=$true, saveat=$4.0)
#ser_t = @benchmark solve($prob, $alg; parallel=$false, saveat=$4.0)


