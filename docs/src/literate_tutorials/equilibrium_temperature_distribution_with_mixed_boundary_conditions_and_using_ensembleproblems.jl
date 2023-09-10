# # Equilibrium Temperature Distribution with Mixed Boundary Conditions and using EnsembleProblems
# For this tutorial, we consider the following problem:
# ```math
# \begin{equation}
# \begin{aligned}
# \grad^2 T &= 0 & \vb x \in \Omega, \\
# \grad T \vdot \vu n &= 0 & \vb x \in \Gamma_1, \\
# T &= 40 & \vb x \in \Gamma_2, \\
# k\grad T \vdot \vu n &= h(T_{\infty} - T) & \vb x \in \Gamma_3, \\
# T &= 70 & \vb x \in \Gamma_4. \\
# \end{aligned}
# \end{equation}
# ```
# This domain $\Omega$ with boundary $\partial\Omega=\Gamma_1\cup\Gamma_2\cup\Gamma_3\cup\Gamma_4$ is shown below.
using CairoMakie #hide
A = (0.0, 0.06) #hide
B = (0.03, 0.06) #hide
F = (0.03, 0.05) #hide
G = (0.05, 0.03) #hide
C = (0.06, 0.03) #hide
D = (0.06, 0.0) #hide
E = (0.0, 0.0) #hide
fig = Figure(fontsize=33) #hide
ax = Axis(fig[1, 1], xlabel="x", ylabel="y") #hide
lines!(ax, [A, E, D], color=:red, linewidth=5) #hide
lines!(ax, [B, F, G, C], color=:blue, linewidth=5) #hide
lines!(ax, [C, D], color=:black, linewidth=5) #hide
lines!(ax, [A, B], color=:magenta, linewidth=5) #hide
text!(ax, [(0.03, 0.001)], text=L"\Gamma_1", fontsize=44) #hide
text!(ax, [(0.055, 0.01)], text=L"\Gamma_2", fontsize=44) #hide
text!(ax, [(0.04, 0.04)], text=L"\Gamma_3", fontsize=44) #hide
text!(ax, [(0.015, 0.053)], text=L"\Gamma_4", fontsize=44) #hide
text!(ax, [(0.001, 0.03)], text=L"\Gamma_1", fontsize=44) #hide
fig #hide

# Let us start by defining the mesh.
using DelaunayTriangulation, FiniteVolumeMethod, CairoMakie
A, B, C, D, E, F, G = (0.0, 0.0),
(0.06, 0.0),
(0.06, 0.03),
(0.05, 0.03),
(0.03, 0.05),
(0.03, 0.06),
(0.0, 0.06)
bn1 = [G, A, B]
bn2 = [B, C]
bn3 = [C, D, E, F]
bn4 = [F, G]
bn = [bn1, bn2, bn3, bn4]
boundary_nodes, points = convert_boundary_points_to_indices(bn)
tri = triangulate(points; boundary_nodes)
refine!(tri; max_area=1e-4get_total_area(tri))
triplot(tri)

#-
mesh = FVMGeometry(tri)

# For the boundary conditions, the parameters that we use are 
# $k = 3$, $h = 20$, and $T_{\infty} = 20$ for thermal conductivity, 
# heat transfer coefficient, and ambient temperature, respectively.
k = 3.0
h = 20.0
T∞ = 20.0
bc1 = (x, y, t, T, p) -> zero(T) # ∇T⋅n=0 
bc2 = (x, y, t, T, p) -> oftype(T, 40.0) # T=40
bc3 = (x, y, t, T, p) -> -p.h * (p.T∞- T) / p.k # k∇T⋅n=h(T∞-T). The minus is since q = -∇T 
bc4 = (x, y, t, T, p) -> oftype(T, 70.0) # T=70
parameters = (nothing, nothing, (h=h, T∞=T∞, k=k), nothing)
BCs = BoundaryConditions(mesh, (bc1, bc2, bc3, bc4),
    (Neumann, Dirichlet, Neumann, Dirichlet);
    parameters)

# Now we can define the actual problem. For the initial condition, 
# which recall is used as an initial guess for steady state problems, 
# let us use an initial condition which ranges from $T=70$ at $y=0.06$
# down to $T=40$ at $y=0$.
diffusion_function = (x, y, t, T, p) -> one(T)
f = (x, y) -> 500y + 40
initial_condition = [f(x, y) for (x, y) in each_point(tri)]
final_time = Inf
prob = FVMProblem(mesh, BCs;
    diffusion_function,
    initial_condition,
    final_time)

#-
steady_prob = SteadyFVMProblem(prob)

# Now we can solve. 
using OrdinaryDiffEq, SteadyStateDiffEq
sol = solve(steady_prob, DynamicSS(Rosenbrock23()))

#-
fig, ax, sc = tricontourf(tri, sol.u, levels=40:70, axis=(xlabel="x", ylabel="y"))
fig
using ReferenceTests #src
@test_reference joinpath(@__DIR__, "../figures", "equilibrium_temperature_distribution_with_mixed_boundary_conditions_and_using_ensembleproblems.png") fig #src

# Let us now suppose we are interested in how the ambient temperature, $T_{\infty}$, 
# affects the temperature distribution. In particular, let us ask the following question:
# > What range of $T_{\infty}$ will allow the temperature at $(0.03, 0.03)$ to be between $50$ and $55$?
# To answer this question, we use an `EnsembleProblem` so that we can solve the problem over many 
# values of $T_{\infty}$ efficiently. For these new problems, it would be 
# a good idea to use a new initial condition given by the solution of the previous problem.
copyto!(prob.initial_condition, sol.u)
using Accessors
T∞_range = LinRange(-100, 100, 101)
ens_prob = EnsembleProblem(steady_prob,
    prob_func=(prob, i, repeat) -> let T∞_range = T∞_range, h = h, k = k
        _prob =
            @set prob.problem.conditions.functions[3].parameters =
                (h=h, T∞=T∞_range[i], k=k) 
        # This way of accessing the parameters is using some internals (specifically,
        # the functions field of conditions is not public API). 
        # A better way could be to e.g. make T∞ mutable initially, e.g. Ref(20),
        # and then mutate it.
        return _prob
    end,
    safetycopy=false)
esol = solve(ens_prob, DynamicSS(Rosenbrock23()); trajectories=length(T∞_range))

# From these results, let us now extract the temperature at $(0.03, 0.03)$. We will use 
# NaturalNeighbours.jl for this.
using NaturalNeighbours
itps = [interpolate(tri, esol[i].u) for i in eachindex(esol)];
itp_vals = [itp(0.03, 0.03; method=Sibson()) for itp in itps]
## If you want piecewise linear interpolation, use either method=Triangle()
## or itp_vals = [pl_interpolate(prob, T, sol.u, 0.03, 0.03) for sol in esol], where 
## T = jump_and_march(tri, (0.03, 0.03)).
using Test #src
_T = jump_and_march(tri, (0.03, 0.03)) #src
_itp_vals = [pl_interpolate(prob, _T, sol.u, 0.03, 0.03) for sol in esol] #src
@test _itp_vals ≈ itp_vals rtol = 1e-4 #src
fig = Figure(fontsize=33)
ax = Axis(fig[1, 1], xlabel=L"T_{\infty}", ylabel=L"T(0.03, 0.03)")
lines!(ax, T∞_range, itp_vals, linewidth=4)
fig

# We see that the temperature at this point seems to increase linearly 
# with $T_{\infty}$. Let us find precisely where this curve 
# meets $T=50$ and $T=55$.
using NonlinearSolve, DataInterpolations
itp = LinearInterpolation(itp_vals, T∞_range)
rootf = (u, p) -> p.itp(u) - p.τ[]
Tthresh = Ref(50.0)
prob = IntervalNonlinearProblem(rootf, (-100.0, 100.0), (itp=itp, τ=Tthresh))
sol50 = solve(prob, ITP())

#-
Tthresh[] = 55.0
sol55 = solve(prob, ITP())

# So, it seems like the answer to our question is $-11.8 \leq T_{\infty} \leq 55$.
# Here is an an animation of the temperature distribution as $T_{\infty}$ varies.
fig = Figure(fontsize=33)
i = Observable(1)
tt = map(i -> L"T_{\infty} = %$(rpad(round(T∞_range[i], digits=3),5,'0'))", i)
u = map(i -> esol.u[i], i)
ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"y",
    title=tt, titlealign=:left)
tricontourf!(ax, tri, u, levels=40:70, extendlow=:auto, extendhigh=:auto)
tightlimits!(ax)
record(fig, joinpath(@__DIR__, "../figures", "temperature_animation.mp4"), eachindex(esol);
    framerate=12) do _i
    i[] = _i
end;

# ![Animation of the temperature distribution](../figures/temperature.mp4)