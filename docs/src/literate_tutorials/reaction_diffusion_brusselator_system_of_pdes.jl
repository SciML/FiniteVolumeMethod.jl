using DisplayAs #hide 
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
# # A Reaction-Diffusion Brusselator System of PDEs 
# In this tutorial, we show how we can solve systems of PDEs. 
# We consider the reaction-diffusion Brusselator system: 
# ```math
# \begin{equation}\label{eq:brusleeq}
# \begin{aligned}
# \pdv{\Phi}{t} &= \frac14\grad^2 \Phi + \Phi^2\Psi - 2\Phi & \vb x \in [0, 1]^2, \\
# \pdv{\Psi}{t} &= \frac14\grad^2 \Psi - \Phi^2\Psi + \Phi & \vb x \in [0, 1]^2,
# \end{aligned}
# \end{equation}
# ```
# Since this is a somewhat contrived example, we will be using the exact 
# solution to define sensible initial and boundary conditions:[^1]
# ```math
# \begin{equation}\label{eq:brusleexct}
# \begin{aligned}
# \Phi(x, y, t) &=\exp(-x-y-t/2), \\
# \Psi(x, y, t) &= \exp(x+y+t/2).
# \end{aligned}
# \end{equation}
# ```
# [^1]: See [Islam, Ali, and Haq (2010)](https://doi.org/10.1016/j.apm.2010.03.028).
# We can use these exact solutions \eqref{eq:brusleexct} to also show how we can mix boundary conditions. 
# We use:
# ```math
# \begin{equation*}
# \begin{aligned}
# \Phi(x, y, 0) &= \exp(-x-y), \\
# \Psi(x, y, 0) &= \exp(x + y), \\
# \pdv{\Phi}{y} &= -\exp(-x-t/2) & y = 0, \\[6pt]
# \Psi &= \exp(x + t/2) & y = 0, \\
# \pdv{\Phi}{x} &= -\exp(-1-y-t/2) & x=1, \\[6pt]
# \pdv{\Psi}{x} &= \exp(1 + y+ t/2) & x=1, \\[6pt] 
# \Phi &= \exp(-1-x-t/2) & y=1, \\
# \pdv{\Psi}{y} &= \exp(1 + x + t/2) & y=1, \\[6pt]
# \pdv{\Phi}{x} &= -\exp(-y-t/2) & x=0, \\[6pt]
# \Psi &= \exp(y + t/2) & x=0. 
# \end{aligned}
# \end{equation*}
# ```
# For implementing these equations, we need to write the Neumann boundary conditions 
# in the forms 
# $\vb q_1 \vdot \vu n = f(\vb x, t)$ and $\vb q_2 \vdot \vu n = f(\vb x, t)$,
# where $\vb q_1$ and $\vb q_2$ are the fluxes for $\Phi$ and $\Psi$, respectively.
# So, we need to rewrite \eqref{eq:brusleeq} in the conservation form; 
# previously, we've also allowed for reaction-diffusion formulations, but unfortunately 
# we do not allow this specification for systems due to some technical limitations. 
# We can write \eqref{eq:brusleeq} in the conservation form as follows:
# ```math 
# \begin{equation}
# \begin{aligned} 
# \pdv{\Phi}{t} + \div\vb q_1 &= S_1, \\
# \pdv{\Psi}{t} + \div\vb q_2 &= S_2, 
# \end{aligned} 
# \end{equation}
# ``` 
# where $\vb q_1 = -\grad\Phi/4$, $S_1 = \Phi^2\Psi - 2\Phi$, 
# $\vb q_2 = -\grad\Psi/4$, and $S_2 = -\Phi^2\Psi + \Phi$. 
# Now that we have these flux functions, let us rewrite our boundary conditions. Remember that 
# $\vu n$ is the outward unit normal, so for example on the bottom boundary we have 
# ```math 
# \vb q_1 \vdot \vu n = -\frac14\grad\Phi \vdot -\vu j = \frac{1}{4}\pdv{\Phi}{y}. 
# ``` 
# The normal vectors are $-\vu j$, $\vu i$, $\vu j$, and $-\vu i$ for the 
# bottom, right, top, and left sides of the square, respectively. So, 
# our boundary become:
# ```math
# \begin{equation*}
# \begin{aligned}
# \Phi(x, y, 0) &= \exp(-x-y), \\
# \Psi(x, y, 0) &= \exp(x + y), \\
# \vb q_1 \vdot \vu n &= -\frac{1}{4}\exp(-x-t/2) & y = 0, \\[6pt]
# \Psi &= \exp(x + t/2) & y = 0, \\
# \vb q_1 \vdot \vu n &= \frac{1}{4}\exp(-1-y-t/2)  & x=1, \\[6pt]
# \vb q_2 \vdot \vu n &= -\frac{1}{4}\exp(1 + y+ t/2) & x=1, \\[6pt] 
# \Phi &= \exp(-1-x-t/2) & y=1, \\
# \vb q_2 \vdot \vu n &= -\frac14\exp(1 + x + t/2) & y=1, \\[6pt]
# \vb q_1 \vdot \vu n &= -\frac14\exp(-y-t/2) & x=0, \\[6pt]
# \Psi &= \exp(y + t/2) & x=0. 
# \end{aligned}
# \end{equation*}
# ```
using FiniteVolumeMethod, DelaunayTriangulation
tri = triangulate_rectangle(0, 1, 0, 1, 100, 100, single_boundary=false)
mesh = FVMGeometry(tri)

# Now we define the boundary conditions. When considering a system of PDEs, 
# you need to define the boundary conditions for each variable separately. 
# The signatures are the same, namely `(x, y, t, u, p) -> Number`, except now 
# `u` is a vector (or `Tuple`) of the solution values for each variable instead 
# of just a scalar. This last point is not relevant here, but you do need to know 
# about it for other problems more generally. So, let us now define the 
# boundary conditions. First, for $\Phi$: 
Φ_bot = (x, y, t, u, p) -> -1 / 4 * exp(-x - t / 2)
Φ_right = (x, y, t, u, p) -> 1 / 4 * exp(-1 - y - t / 2)
Φ_top = (x, y, t, u, p) -> exp(-1 - x - t / 2)
Φ_left = (x, y, t, u, p) -> -1 / 4 * exp(-y - t / 2)
Φ_bc_fncs = (Φ_bot, Φ_right, Φ_top, Φ_left)
Φ_bc_types = (Neumann, Neumann, Dirichlet, Neumann)
Φ_BCs = BoundaryConditions(mesh, Φ_bc_fncs, Φ_bc_types)

# Now, for $\Psi$: 
Ψ_bot = (x, y, t, u, p) -> exp(x + t / 2)
Ψ_right = (x, y, t, u, p) -> -1 / 4 * exp(1 + y + t / 2)
Ψ_top = (x, y, t, u, p) -> -1 / 4 * exp(1 + x + t / 2)
Ψ_left = (x, y, t, u, p) -> exp(y + t / 2)
Ψ_bc_fncs = (Ψ_bot, Ψ_right, Ψ_top, Ψ_left)
Ψ_bc_types = (Dirichlet, Neumann, Neumann, Dirichlet)
Ψ_BCs = BoundaryConditions(mesh, Ψ_bc_fncs, Ψ_bc_types)

# Now we need to define the actual problems. Let us first define the flux 
# and source functions, remembering that the variables get replaced with 
# linear approximants. The flux functions also now take `Tuple`s for $\alpha$, 
# $\beta$, and $\gamma$, where the $i$th element of the `Tuple` refers to the 
# $i$th variable. Similarly, the source function takes a `Tuple` of the variables 
# in the `u` argument.
Φ_q = (x, y, t, α, β, γ, p) -> (-α[1] / 4, -β[1] / 4)
Ψ_q = (x, y, t, α, β, γ, p) -> (-α[2] / 4, -β[2] / 4)
Φ_S = (x, y, t, (Φ, Ψ), p) -> Φ^2 * Ψ - 2Φ
Ψ_S = (x, y, t, (Φ, Ψ), p) -> -Φ^2 * Ψ + Φ

# Now we define the initial conditions. 
Φ_exact = (x, y, t) -> exp(-x - y - t / 2)
Ψ_exact = (x, y, t) -> exp(x + y + t / 2)
Φ₀ = [Φ_exact(x, y, 0) for (x, y) in each_point(tri)]
Ψ₀ = [Ψ_exact(x, y, 0) for (x, y) in each_point(tri)];

# Next, we can define the `FVMProblem`s for each variable. 
Φ_prob = FVMProblem(mesh, Φ_BCs; flux_function=Φ_q, source_function=Φ_S,
    initial_condition=Φ₀, final_time=5.0)

#-
Ψ_prob = FVMProblem(mesh, Ψ_BCs; flux_function=Ψ_q, source_function=Ψ_S,
    initial_condition=Ψ₀, final_time=5.0)

# Finally, the `FVMSystem` is constructed by these two problems: 
system = FVMSystem(Φ_prob, Ψ_prob)

# We can now solve the problem just as we've done previously. 
using OrdinaryDiffEq, LinearSolve
sol = solve(system, TRBDF2(linsolve=KLUFactorization()), saveat=1.0)
sol |> tc #hide

# For this solution, note that the `u` values are matrices. For example: 
sol.u[3]
sol.u[3] |> tc #hide

# The `i`th row is the `i`th variable, so 
sol.u[3][1, :]
sol.u[3][1, :] |> tc #hide

# are the value of $\Phi$ at the third time, and similarly `sol.u[3][2, :]` 
# are the values of $\Psi$ at the third time. We can visualise the solutions as follows:
using CairoMakie
fig = Figure(fontsize=38)
for i in eachindex(sol)
    ax1 = Axis(fig[1, i], xlabel=L"x", ylabel=L"y",
        width=400, height=400,
        title=L"\Phi: t = %$(sol.t[i])", titlealign=:left)
    ax2 = Axis(fig[2, i], xlabel=L"x", ylabel=L"y",
        width=400, height=400,
        title=L"\Psi: t = %$(sol.t[i])", titlealign=:left)
    tricontourf!(ax1, tri, sol[i][1, :], levels=0:0.1:1, colormap=:matter)
    tricontourf!(ax2, tri, sol[i][2, :], levels=1:10:100, colormap=:matter)
end
resize_to_layout!(fig)
fig
using ReferenceTests #src
@test_reference joinpath(@__DIR__, "../figures", "reaction_diffusion_brusselator_system_of_pdes.png") fig #src

x = getx.(get_points(tri)) #src
y = gety.(get_points(tri)) #src
for i in eachindex(sol) #src
    ax3 = Axis(fig[3, i], xlabel=L"x", ylabel=L"y", #src
        width=400, height=400, #src
        title=L"Exact $\Phi: t = %$(sol.t[i])$", titlealign=:left) #src
    ax4 = Axis(fig[4, i], xlabel=L"x", ylabel=L"y", #src
        width=400, height=400, #src
        title=L"Exact $\Psi: t = %$(sol.t[i])$", titlealign=:left) #src
    tricontourf!(ax3, tri, Φ_exact.(x, y, sol.t[i]), levels=0:0.1:1, colormap=:matter) #src
    tricontourf!(ax4, tri, Ψ_exact.(x, y, sol.t[i]), levels=1:10:100, colormap=:matter) #src
end  #src
resize_to_layout!(fig) #src
fig #src
@test_reference joinpath(@__DIR__, "../figures", "reaction_diffusion_brusselator_system_of_pdes_exact_comparisons.png") fig #src