# # A Cylic Competition Model 
# This tutorial considers the following system:
# ```math
# \begin{equation}
# \begin{aligned}
# \pdv{u}{t} &= D_u\grad^2 u + u(1-u-av-bw), \\
# \pdv{v}{t} &= D_v\grad^2 v + v(1-bu-v-aw), \\
# \pdv{w}{t} &= D_w\grad^2 w + w(1-au-bv-w),
# \end{aligned}
# \end{equation}
# ```
# with homogeneous boundary conditions on a rectangle. We start by defining 
# the mesh.
using DelaunayTriangulation, FiniteVolumeMethod
a, b, c, d = 0.0, 500.0, 0.0, 500.0
tri = triangulate_rectangle(a, b, c, d, 500, 500, single_boundary=true)
mesh = FVMGeometry(tri)

# For the boundary conditions, remember that for systems of PDEs 
# we need to provide a `BoundaryConditions` for each equation. 
bc_u = (x, y, t, (u, v, w), p) -> zero(u)
bc_v = (x, y, t, (u, v, w), p) -> zero(v)
bc_w = (x, y, t, (u, v, w), p) -> zero(w)
BC_u = BoundaryConditions(mesh, bc_u, Neumann)
BC_v = BoundaryConditions(mesh, bc_v, Neumann)
BC_w = BoundaryConditions(mesh, bc_w, Neumann)

# Now we define the problem. The parameters we use are: 
Du, Dv, Dw = 2.0, 1 / 2, 1 / 2
a, b = 0.8, 1.9

# We cannot use the diffusion formulation for systems, so we instead 
# define the problem in terms of the flux functions $\vb q_1 = -\grad u$, 
# $\vb q_2 = -\grad v$, and $\vb q_3 = -\grad w$. We use 
# an initial condition that leads to some nice patterns.
flux_u = (x, y, t, α, β, γ, p) -> (-α[1] * p.Du, -β[1] * p.Du)
flux_v = (x, y, t, α, β, γ, p) -> (-α[2] * p.Dv, -β[2] * p.Dv)
flux_w = (x, y, t, α, β, γ, p) -> (-α[3] * p.Dw, -β[3] * p.Dw)
source_u = (x, y, t, (u, v, w), p) -> u * (1 - u - p.a * v - p.b * w)
source_v = (x, y, t, (u, v, w), p) -> v * (1 - p.b * u - v - p.a * w)
source_w = (x, y, t, (u, v, w), p) -> w * (1 - p.a * u - p.b * v - w)
flux_u_parameters = (Du=Du,)
flux_v_parameters = (Dv=Dv,)
flux_w_parameters = (Dw=Dw,)
source_u_parameters = source_v_parameters = source_w_parameters = (a=a, b=b)
final_time = 1e3
uvw_init = (x, y) -> sech(sqrt((x - 250)^2 + (y - 250)^2))
ic_u = ic_v = ic_w = [uvw_init(x, y) for (x, y) in each_point(tri)]
prob_u = FVMProblem(mesh, BC_u;
    flux_function=flux_u, flux_parameters=flux_u_parameters,
    source_function=source_u, source_parameters=source_u_parameters,
    initial_condition=ic_u, final_time=final_time)
prob_v = FVMProblem(mesh, BC_v;
    flux_function=flux_v, flux_parameters=flux_v_parameters,
    source_function=source_v, source_parameters=source_v_parameters,
    initial_condition=ic_v, final_time=final_time)
prob_w = FVMProblem(mesh, BC_w;
    flux_function=flux_w, flux_parameters=flux_w_parameters,
    source_function=source_w, source_parameters=source_w_parameters,
    initial_condition=ic_w, final_time=final_time)
sys = FVMSystem(prob_u, prob_w, prob_w)

# We can now solve the system. 
using OrdinaryDiffEq, LinearSolve, CairoMakie
sol = solve(sys, TRBDF2(linsolve=KLUFactorization()), saveat=1.0)

x = LinRange(0, 500, 500)
y = LinRange(0, 500, 500)
i = Observable(1)
u = map(i -> reshape(sol.u[i][1, :] .^ 2 + sol.u[i][2, :] .^ 2 + sol.u[i][3, :] .^ 2, 500, 500), i)
fig, ax, sc = heatmap(x, y, u, colorrange=(0, 1))
hidedecorations!(ax)
tightlimits!(ax)
record(fig, joinpath(@__DIR__, "../figures", "cyclic_patterns.mp4"), eachindex(sol);
    framerate=60) do _i
    i[] = _i
end;
# ```@raw html
# <figure>
#     <img src='../figures/cyclic_patterns.mp4', alt='Animation of the cyclic patterns'><br>
# </figure>
# ```
