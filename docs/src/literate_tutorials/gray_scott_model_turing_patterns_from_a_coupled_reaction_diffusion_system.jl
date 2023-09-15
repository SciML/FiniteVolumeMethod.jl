# # Gray-Scott Model: Turing Patterns from a Coupled Reaction-Diffusion System
#
# In this tutorial, we explore some pattern formation from the
# Gray-Scott model:
# ```math
# \begin{equation}
# \begin{aligned}
# \pdv{u}{t} &= \varepsilon_1\grad^2u+b(1-u)-uv^2, \\
# \pdv{v}{t} &= \varepsilon_2\grad^2 v - dv+uv^2,
# \end{aligned}
# \end{equation}
# ```
# where $u$ and $v$ are the concentrations of two chemical species. The
# initial conditions we use are:
# ```math
# \begin{align*}
# u(x, y, 0) &= 1 -\exp\left[-80\left(x^2 + y^2\right)\right], \\
# v(x, y, 0) &= \exp\left[-80\left(x^2+y^2\right)\right].
# \end{align*}
# ```
# The domain we use is $[-1, 1]^2$, and we use
# zero flux boundary conditions.
using FiniteVolumeMethod, DelaunayTriangulation
tri = triangulate_rectangle(-1, 1, -1, 1, 200, 200, single_boundary=true)
mesh = FVMGeometry(tri)

#-
bc = (x, y, t, (u, v), p) -> zero(u) * zero(v)
u_BCs = BoundaryConditions(mesh, bc, Neumann)
v_BCs = BoundaryConditions(mesh, bc, Neumann)

#-
ε₁ = 0.00002
ε₂ = 0.00001
b = 0.04
d = 0.1
u_q = (x, y, t, α, β, γ, _ε₁) -> (-α[1] * _ε₁, -β[1] * _ε₁)
v_q = (x, y, t, α, β, γ, _ε₂) -> (-α[2] * _ε₂, -β[2] * _ε₂)
u_S = (x, y, t, (u, v), _b) -> _b * (1 - u) - u * v^2
v_S = (x, y, t, (u, v), _d) -> -_d * v + u * v^2
u_qp = ε₁
v_qp = ε₂
u_Sp = b
v_Sp = d
u_icf = (x, y) -> 1 - exp(-80 * (x^2 + y^2))
v_icf = (x, y) -> exp(-80 * (x^ 2 + y^2))
u_ic = [u_icf(x, y) for (x, y) in each_point(tri)]
v_ic = [v_icf(x, y) for (x, y) in each_point(tri)]
u_prob = FVMProblem(mesh, u_BCs;
    flux_function=u_q, flux_parameters=u_qp,
    source_function=u_S, source_parameters=u_Sp,
    initial_condition=u_ic, final_time=6000.0)
v_prob = FVMProblem(mesh, v_BCs;
    flux_function=v_q, flux_parameters=v_qp,
    source_function=v_S, source_parameters=v_Sp,
    initial_condition=v_ic, final_time=6000.0)
prob = FVMSystem(u_prob, v_prob)

# Now that we have our system, we can solve.
using OrdinaryDiffEq, Sundials
sol = solve(prob, CVODE_BDF(linear_solver=:GMRES), saveat=10.0, parallel=Val(false)) 

# Here is an animation of the solution, looking only at the $v$ variable.
using CairoMakie
fig = Figure(fontsize=33)
ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"y")
tightlimits!(ax)
i = Observable(1)
u = map(i -> reshape(sol.u[i][2, :], 200, 200), i)
x = LinRange(-1, 1, 200)
y = LinRange(-1, 1, 200)
heatmap!(ax, x, y, u, colorrange=(0.0, 0.4))
hidedecorations!(ax)
record(fig, joinpath(@__DIR__, "../figures", "gray_scott_patterns.mp4"), eachindex(sol);
    framerate=60) do _i
    i[] = _i
end

# ![Animation of the Gray-Scott model](../figures/gray_scott_patterns.mp4)