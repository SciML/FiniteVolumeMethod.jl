using DelimitedFiles
wave_pore_bnd = readdlm("C:/Users/licer/.julia/dev/TissueMechanics/New_Data_9112022/matlab/geometric/boundaries/IMAGE14PORE1.dat")
λ = 0.5
D = 150.0
u₀ = 0.25
initial_time = 4.0
final_time = 28.0
target_n = 5_000
plot_times = initial_time:3:final_time
x = wave_pore_bnd[:, 1]
y = wave_pore_bnd[:, 2]
x = x .- minimum(x)
y = y .- minimum(y)
r = 7.97576904296875
T, adj, adj2v, DG, pts, BN = generate_mesh(x, y, r; gmsh_path=GMSH_PATH)
msh = FVMGeometry(T, adj, adj2v, DG, pts, BN)
functions = (x, y, t, u, p) -> p[1] * u * (1 - u)
type = :dudt
params = [[λ]]
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
flux_parameters = [D]
reaction_parameters = [λ]
initial_condition = zeros(num_points(pts))
initial_condition[BC.boundary_node_vector[1]] .= u₀
prob = FVMProblem(;
    mesh=msh,
    boundary_conditions=BC,
    (flux!)=flux_fnc!,
    reaction=Rf,
    flux_parameters=flux_parameters,
    reaction_parameters=reaction_parameters,
    initial_time=initial_time,
    final_time=final_time,
    initial_condition=initial_condition,
    solver=Rosenbrock23(linsolve=KLUFactorization()))
sol = solve(prob; saveat=plot_times)
