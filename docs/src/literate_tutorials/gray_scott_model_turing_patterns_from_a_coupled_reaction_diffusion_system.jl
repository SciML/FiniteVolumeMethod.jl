# # Gray-Scott Model: Turing Patterns from a Coupled Reaction-Diffusion System
# In this tutorial, we explore some pattern formation from the 
# Gray-Scott model:
# ```math
# \begin{equation}
# \begin{aligned}
# \pdv{u}{t} &= \varepsilon_1\grad^2u+b(1-u)-uv^2, \\
# \pdv{v}{t} &= \varepsilon_2\grad v - dv+uv^2,
# \end{aligned}
# \end{equation}
# ```
# where $u$ and $v$ are the concentrations of two chemical species. The 
# initial conditions we use are:
# ```math
# \begin{align*}
# u(x, y) &= 1 -\exp\left\{-80\left[\left(x+\frac12\right)^2 + \left(y + \frac15\right)^2\right]\right\}, \\
# v(x, y) &= \exp\left\{-80\left[\left(x-\frac12\right)^2+\left(y-\frac15\right)^2\right]\right\}.
# \end{align*}
# ```
# The domain we use is $[-1, 1]^2$, and we use 
# zero flux boundary conditions.
using FiniteVolumeMethod, DelaunayTriangulation
tri = triangulate_rectangle(-1, 1, -1, 1, 20, 20, single_boundary=true)
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
u_icf = (x, y) -> 1 - exp(-80 * ((x)^2 + (y) .^ 2))
v_icf = (x, y) -> exp(-80 * ((x) .^ 2 + (y) .^ 2))
u_ic = [u_icf(x, y) for (x, y) in each_point(tri)]
v_ic = [v_icf(x, y) for (x, y) in each_point(tri)]
u_prob = FVMProblem(mesh, u_BCs;
    flux_function=u_q, flux_parameters=u_qp,
    source_function=u_S, source_parameters=u_Sp,
    initial_condition=u_ic, final_time=3500.0)
v_prob = FVMProblem(mesh, v_BCs;
    flux_function=v_q, flux_parameters=v_qp,
    source_function=v_S, source_parameters=v_Sp,
    initial_condition=v_ic, final_time=3500.0)
prob = FVMSystem(u_prob, v_prob)

# Now that we have our systme, we can solve. 
using OrdinaryDiffEq, LinearSolve
sol = solve(prob, TRBDF2(linsolve=KLUFactorization()), saveat=10.0)

alg = TRBDF2(linsolve=KLUFactorization())
@benchmark solve($prob, $alg, saveat=$10.0)

du = zero(prob.initial_condition)
u = prob.initial_condition 
p = FiniteVolumeMethod.get_multithreading_vectors(prob)
t = 0.0
@benchmark $FiniteVolumeMethod.fvm_eqs!($du,$u,$p,$t)

tricontourf(tri, sol.u[10][2, :])

julia> @benchmark $FiniteVolumeMethod.fvm_eqs!($du,$u,$p,$t)
BenchmarkTools.Trial: 1685 samples with 1 evaluation.
 Range (min … max):  211.000 μs … 333.495 ms  ┊ GC (min … max):  0.00% … 99.36%
 Time  (median):       2.174 ms               ┊ GC (median):     0.00%
 Time  (mean ± σ):     2.957 ms ±  15.922 ms  ┊ GC (mean ± σ):  26.27% ±  4.84%

                                            ▂▅▅█▄▄▂
  ▂▂▂▂▂▁▂▂▁▁▁▁▁▁▁▁▁▂▁▁▁▂▁▁▁▁▁▂▁▁▁▁▂▂▂▂▂▃▃▄▅▇████████▆▅▄▄▃▃▃▂▂▂▂ ▃
  211 μs           Histogram: frequency by time         2.84 ms <

 Memory estimate: 9.52 MiB, allocs estimate: 33297.