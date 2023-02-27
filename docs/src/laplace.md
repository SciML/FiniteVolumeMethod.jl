# Example VIII: Laplace's equation

Now we consider Laplace's equaton. This is a steady problem, whereas all previous examples thus far have been unsteady. The problem we consider is 

```math
\begin{equation*}
\begin{array}{rcll}
\boldsymbol{\nabla}^2 u(x, y) & = & 0 & 0 < x, y < \pi, \\
u(x, 0) & = & \sinh(x) & 0 < x < \pi,
u(x, \pi) & = & -\sinh(x) & 0 < x < \pi, 
u(0, y) & = & 0 & 0 < y < \pi, \\
u(\pi, y) & = & \sinh(\pi)\cos(y) & 0 < x < \pi,
\end{array}
\end{equation*}
```

which has exact solution $u(x, y) = \sinh(x)\cos(y)$. 

For a steady problem, rather than defining problems of the form

```math 
\begin{equation*}
\dfrac{\partial u}{\partial t} + \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{q} = R,
\end{equation*}
```

we are solving problems with $\partial u/\partial t = 0$ so that 

```math 
\begin{equation*}
\boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{q} = R,
\end{equation*}
```

For the reaction-diffusion formula, this is instead given by $0 = \boldsymbol\nabla[D\boldsymbol\nabla u] + R$. Moreover, the initial condition that we provide is now actually used as an initial estimate in the nonlinear solver that computes the steady state (defined via NonlinearSolve.jl). Lastly, the final time that we provide is replaced with infinity. With this in mind, the code for solving the above problem is given below.

```julia
using FiniteVolumeMethod
using CairoMakie
using OrdinaryDiffEq
using LinearSolve
using SteadyStateDiffEq 
using Krylov

## Define the problem
a, b = 0.0, π
c, d = 0.0, π
tri = generate_mesh(a, b, c, d, 0.2; gmsh_path=GMSH_PATH, single_boundary=false)
mesh = FVMGeometry(tri)
bc1 = (x, y, t, u, p) -> sinh(x)
bc2 = (x, y, t, u, p) -> sinh(π) * cos(y)
bc3 = (x, y, t, u, p) -> -sinh(x)
bc4 = (x, y, t, u, p) -> zero(u)
types = [:D, :D, :D, :D]
BCs = BoundaryConditions(mesh, [bc1, bc2, bc3, bc4], types)
diffusion_function = (x, y, t, u, p) -> one(u)
final_time = Inf
initial_guess = zeros(num_points(tri))
for (i, f) in pairs((bc1, bc2, bc3, bc4)) 
    bn = get_boundary_nodes(tri, i)
    for j in bn
        p = get_point(tri, j)
        x, y = getxy(p)
        initial_guess[j] = f(x, y, Inf, 0.0, nothing)
    end
end
prob = FVMProblem(mesh, BCs;
    diffusion_function,
    final_time,
    initial_condition=initial_guess,
    steady=true)

## Solve with a steady state algorithm
alg = DynamicSS(TRBDF2(linsolve=KrylovJL_GMRES()))
sol = solve(prob, alg, parallel=true)

## Plot 
fig = Figure(fontsize=38)
ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"y")
pt_mat = Matrix(get_points(tri)')
T_mat = [T[j] for T in each_triangle(tri), j in 1:3]
msh = mesh!(ax, pt_mat, T_mat, color=sol, colorrange=(-15, 15))
Colorbar(fig[1, 2], msh)
```

![Circle mean exit time](https://github.com/DanielVandH/FiniteVolumeMethod.jl/blob/main/test/figures/laplace_equation.png?raw=true)