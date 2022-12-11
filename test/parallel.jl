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
r = 0.02
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
@test_throws "The flux vector" solve(prob, TRBDF2(linsolve=KLUFactorization()); parallel=true)

prob = FVMProblem(mesh, BCs; iip_flux=false,
    diffusion_function=D, reaction_function=R,
    initial_condition=u₀, final_time
)
sol_par = solve(prob, TRBDF2(linsolve=KLUFactorization(), autodiff=true); parallel=true, saveat=0.05)
sol_ser = solve(prob, TRBDF2(linsolve=KLUFactorization(), autodiff=true); parallel=false, saveat=0.05)
@test sol_par.u ≈ sol_ser.u

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
prob = FVMProblem(mesh, BCs; iip_flux=false, diffusion_function=D, initial_condition=u₀, final_time)
sol_par = solve(prob, TRBDF2(linsolve=KLUFactorization(), autodiff=true); parallel=true, saveat=0.05)
sol_ser = solve(prob, TRBDF2(linsolve=KLUFactorization(), autodiff=true); parallel=false, saveat=0.05)
@test sol_par.u ≈ sol_ser.u

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
alg = FBDF(linsolve=UMFPACKFactorization())
sol_par = solve(prob, alg; parallel=true, saveat=0.05)
sol_ser = solve(prob, alg; parallel=false, saveat=0.05)
@test sol_par.u ≈ sol_ser.u

