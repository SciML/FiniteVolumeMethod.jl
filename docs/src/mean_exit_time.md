# Example IX: Mean exit time problems

In this example, we will investigate some simple mean exit time problems. In particular, the aim is to reproduce some of the figures from the numerical solutions in my other work on mean exit time at https://iopscience.iop.org/article/10.1088/1367-2630/abe60d and https://iopscience.iop.org/article/10.1088/1751-8121/ac4a1d.

Fistly, mean exit time problems with linear diffusion can be defined according to

```math 
\begin{equation*}
\begin{array}{rcll}
D\boldsymbol{\nabla}^2 T(\boldsymbol x) & = & -1 & \boldsymbol x \in \Omega, \\
T(\boldsymbol x) & = & 0 & \boldsymbol x \in \partial\Omega,
\end{array}
\end{equation*}
```

for some diffusivity $D$. $T(\boldsymbol x)$ is the mean exit time at $\boldsymbol x$, meaning the average amount of time it would take a particle starting at $\boldsymbol x$ to exit the domain through $\partial\Omega$. For this interpretation of $T$, we are letting $D = \mathcal P\delta^2/(4\tau)$, where $\delta>0$ is the step length of the particle, $\tau > 0$ is the duration between steps, and $\mathcal P \in [0, 1]$ is the probability that the particle actually moves at a given time step.

To define the problem with our interface, remember that we use the PDE (using the reaction-diffusion formulation rather than the formulation with $\boldsymbol q$)

```math 
\begin{equation*}
\dfrac{\partial T}{\partial t} = \boldsymbol{\nabla} \boldsymbol{\cdot} \left[ D(x, y, t, T)\boldsymbol{\nabla} T(x, y, t)\right] + R(x, y, t, T).
\end{equation*}
```

Since this is a steady problem, $\partial T/\partial t = 0$, meaning

```math
\begin{equation*}
\boldsymbol{\nabla} \boldsymbol{\cdot} \left[ D(x, y, t, T)\boldsymbol{\nabla} T(x, y, t)\right] = -R(x, y, t, T).
\end{equation*}
```

Therefore, we should define a problem with $D(x, y, t, T) = D$ and $R(x, y, t, T) = 1$. Let's now give the examples.

## Circle

Let's first solve the problem on a unit circle. In this case, the exact solution is $T(x, y) = (1 - r^2)/(4D)$, $r = \sqrt{x^2 + y^2}$. For the initial guess for $T$, we will just set $T=1$ everywhere and $T=0$ on the boundary.

```julia
using FiniteVolumeMethod
using DelaunayTriangulation
using CairoMakie
using OrdinaryDiffEq
using SteadyStateDiffEq
using Test
using LinearAlgebra
using StatsBase

R = 1.0
θ = LinRange(0, 2π, 100)
x = R .* cos.(θ)
y = R .* sin.(θ)
tri = generate_mesh(x, y, 0.2; gmsh_path=GMSH_PATH)
mesh = FVMGeometry(tri)
bc = (x, y, t, u, p) -> zero(u)
type = :D
BCs = BoundaryConditions(mesh, bc, type)
D = 2.5e-5
initial_guess = ones(num_points(tri))
initial_guess[get_boundary_nodes(tri)] .= 0.0
diffusion_function = (x, y, t, u, D) -> D
diffusion_parameters = D
reaction = (x, y, t, u, p) -> one(u)
final_time = Inf # doesn't matter in this case, but we still need to provide it
prob = FVMProblem(mesh, BCs;
    diffusion_function,
    diffusion_parameters,
    reaction_function=reaction,
    final_time=final_time,
    initial_condition=initial_guess,
    steady=true)

# Solve 
alg = DynamicSS(Rosenbrock23())
sol = solve(prob, alg)

# Visualise 
fig = Figure(fontsize=38)
ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"y")
pt_mat = Matrix(get_points(tri)')
T_mat = [T[j] for T in each_triangle(tri), j in 1:3]
msh = mesh!(ax, pt_mat, T_mat, color=sol, colorrange=(0, 10000))
Colorbar(fig[1, 2], msh)

# Test 
exact = [(R^2 - norm(p)^2)/(4D) for p in each_point(tri)]
error_fnc = (fvm_sol, exact_sol) -> 100abs(fvm_sol - exact_sol)/maximum(exact)
@test median(error_fnc.(exact, sol)) < 0.2
```

![Circle mean exit time](https://github.com/DanielVandH/FiniteVolumeMethod.jl/blob/main/test/figures/circle_mean_exit_time.png?raw=true)

We can now extend this problem by considering a perturbed domain. In particular, we consider a circle with boundary defined by $\mathcal R(\theta) = 1 + \varepsilon g(\theta)$ for a small perturbation $\varepsilon = 1/20$ and $g(\theta) =\sin(3\theta) + \cos(5\theta) - \sin(\theta)$. For the initial guess, we will use the solution from the unperturbed problem.

```julia
using FiniteVolumeMethod
using DelaunayTriangulation
using CairoMakie
using OrdinaryDiffEq
using SteadyStateDiffEq
using LinearSolve
using FastLapackInterface

R = 1.0
θ = LinRange(0, 2π, 100)
ε = 1/20
g = θ -> sin(3θ) + cos(5θ) - sin(θ)
R_bnd = 1 .+ ε .* g.(θ)
x = R_bnd .* cos.(θ)
y = R_bnd .* sin.(θ)
tri = generate_mesh(x, y, 0.2; gmsh_path=GMSH_PATH)
mesh = FVMGeometry(tri)
bc = (x, y, t, u, p) -> zero(u)
type = :D
BCs = BoundaryConditions(mesh, bc, type)
D = 2.5e-5
initial_guess = [(R^2 - norm(p)^2)/(4D) for p in each_point(tri)]
initial_guess[get_boundary_nodes(tri)] .= 0.0
diffusion_function = (x, y, t, u, D) -> D
diffusion_parameters = D
reaction = (x, y, t, u, p) -> one(u)
final_time = Inf # doesn't matter in this case, but we still need to provide it
prob = FVMProblem(mesh, BCs;
    diffusion_function,
    diffusion_parameters,
    reaction_function=reaction,
    final_time=final_time,
    initial_condition=initial_guess,
    steady=true)
alg = DynamicSS(TRBDF2(linsolve=FastLUFactorization()))
sol = solve(prob, alg, parallel = true)
fig = Figure(fontsize=38)
ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"y")
pt_mat = Matrix(get_points(tri)')
T_mat = [T[j] for T in each_triangle(tri), j in 1:3]
msh = mesh!(ax, pt_mat, T_mat, color=sol, colorrange=(0, 10000))
Colorbar(fig[1, 2], msh)
```

![Perturbed circle mean exit time](https://github.com/DanielVandH/FiniteVolumeMethod.jl/blob/main/test/figures/perturbed_circle_mean_exit_time.png?raw=true)

## Ellipse

Now let's consider the same problem but on an ellipse. Here, for an unperturbed ellipse with width $2a>0$ and height $2b>0$ ($a>b$), the solution is

```math 
T(x, y) = \frac{a^2b^2}{2D(a^2+b^2)}\left[1-\frac{x^2}{y^2}-\frac{y^2}{b^2}\right].
```

The code below solves the unperturbed problem.

```julia
using FiniteVolumeMethod
using DelaunayTriangulation
using CairoMakie
using OrdinaryDiffEq
using SteadyStateDiffEq
using Test
using LinearAlgebra
using StatsBase
using LinearSolve
using Krylov

a = 2.0
b = 1.0
θ = LinRange(0, 2π, 100)
x = a * cos.(θ)
y = b * sin.(θ)
tri = generate_mesh(x, y, 0.2; gmsh_path=GMSH_PATH)
mesh = FVMGeometry(tri)
bc = (x, y, t, u, p) -> zero(u)
type = :D
BCs = BoundaryConditions(mesh, bc, type)
D = 2.5e-5
initial_guess = ones(num_points(tri))
initial_guess[get_boundary_nodes(tri)] .= 0.0
diffusion_function = (x, y, t, u, D) -> D
diffusion_parameters = D
reaction = (x, y, t, u, p) -> one(u)
final_time = Inf
prob = FVMProblem(mesh, BCs;
    diffusion_function,
    diffusion_parameters,
    reaction_function=reaction,
    final_time=final_time,
    initial_condition=initial_guess,
    steady=true)
sol = solve(prob, DynamicSS(TRBDF2(linsolve=KrylovJL_GMRES())))
fig = Figure(fontsize=38)
ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"y", width=600, height=300)
pt_mat = Matrix(get_points(tri)')
T_mat = [T[j] for T in each_triangle(tri), j in 1:3]
msh = mesh!(ax, pt_mat, T_mat, color=sol, colorrange=(0, 16000))
Colorbar(fig[1, 2], msh)
resize_to_layout!(fig)
exact = [a^2 * b^2 / (2 * D * (a^2 + b^2)) * (1 - x^2 / a^2 - y^2 / b^2) for (x, y) in each_point(tri)]
error_fnc = (fvm_sol, exact_sol) -> 100abs(fvm_sol - exact_sol) / maximum(exact)
@test median(error_fnc.(exact, sol)) < 0.3
```

![Ellipse mean exit time](https://github.com/DanielVandH/FiniteVolumeMethod.jl/blob/main/test/figures/ellipse_mean_exit_time.png?raw=true)

Now we consider a perturbed ellipse. In particular, we take $x = a(1 + \varepsilon g(\theta))\cos(\theta)$ and $y = b(1 + \varepsilon h(\theta))\sin(\theta)$, with $\varepsilon 1/20$, $g(\theta) = \sin(3\theta) + \cos(5\theta) - \sin(\theta)$, and $h(\theta) = \cos(3\theta) + \sin(5\theta) - \cos(\theta)$. We will start our initial guess at the zero vector.

```julia
using FiniteVolumeMethod
using DelaunayTriangulation
using CairoMakie
using OrdinaryDiffEq
using SteadyStateDiffEq
using Test
using LinearSolve
using Krylov

a = 2.0
b = 1.0
θ = LinRange(0, 2π, 100)
ε = 1 / 20
g = θ -> sin(3θ) + cos(5θ) - sin(θ)
h = θ -> cos(3θ) + sin(5θ) - cos(θ)
x = a * (1 .+ ε .* g.(θ)) .* cos.(θ)
y = b * (1 .+ ε .* h.(θ)) .* sin.(θ)
tri = generate_mesh(x, y, 0.2; gmsh_path=GMSH_PATH)
mesh = FVMGeometry(tri)
bc = (x, y, t, u, p) -> zero(u)
type = :D
BCs = BoundaryConditions(mesh, bc, type)
D = 2.5e-5
initial_guess = zeros(num_points(tri))
diffusion_function = (x, y, t, u, D) -> D
diffusion_parameters = D
reaction = (x, y, t, u, p) -> one(u)
final_time = Inf
prob = FVMProblem(mesh, BCs;
    diffusion_function,
    diffusion_parameters,
    reaction_function=reaction,
    final_time=final_time,
    initial_condition=initial_guess,
    steady=true)
sol = solve(prob, DynamicSS(TRBDF2(linsolve=KrylovJL_GMRES())))
fig = Figure(fontsize=38)
ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"y", width=600, height=300)
pt_mat = Matrix(get_points(tri)')
T_mat = [T[j] for T in each_triangle(tri), j in 1:3]
msh = mesh!(ax, pt_mat, T_mat, color=sol, colorrange=(0, 16000))
Colorbar(fig[1, 2], msh)
resize_to_layout!(fig)
```

![Perturbed ellipse mean exit time](https://github.com/DanielVandH/FiniteVolumeMethod.jl/blob/main/test/figures/perturbed_ellipse_mean_exit_time.png?raw=true)

## Annulus

Now we consider mean exit time on an annulus. The paper https://iopscience.iop.org/article/10.1088/1751-8121/ac4a1d also considers heterogeneous annuli, but we will not do that here. We will consider a reflecting inner boundary condition with an absorbing outer boundary condition, so that $\boldsymbol{\nabla}^2 T = -1/D$ in the annulus, $\boldsymbol{\nabla} T \vdot \hat{\boldsymbol n} = 0$ on the inner ring, and $T = 0$ on the outer ring. The annulus we consider is $R_1 < r < R_2$ with $R_1 = 2$ and $R_2 = 3$, which corresponds to an exact solution $T(x, y) = (R_2^2 - r^2)/(4D) + R_1^2\log(r/R_2)/(2D)$, where $r = \sqrt{x^2 + y^2}$.

```julia 
using FiniteVolumeMethod
using DelaunayTriangulation
using CairoMakie
using OrdinaryDiffEq
using SteadyStateDiffEq
using Test
using LinearAlgebra
using StatsBase
using LinearSolve
using Krylov

R₁ = 2.0
R₂ = 3.0
θ = LinRange(0, 2π, 250)
x = [
    [R₂ .* cos.(θ)],
    [reverse(R₁ .* cos.(θ))] # inner boundaries are clockwise
]
y = [
    [R₂ .* sin.(θ)],
    [reverse(R₁ .* sin.(θ))] # inner boundaries are clockwise
]
tri = generate_mesh(x, y, 0.2; gmsh_path=GMSH_PATH)
mesh = FVMGeometry(tri)
outer_bc = (x, y, t, u, p) -> zero(u)
inner_bc = (x, y, t, u, p) -> zero(u)
type = [:D, :N]
BCs = BoundaryConditions(mesh, [outer_bc, inner_bc], type)
D = 6.25e-4
initial_guess = zeros(num_points(tri))
diffusion_function = (x, y, t, u, D) -> D
diffusion_parameters = D
reaction = (x, y, t, u, p) -> one(u)
final_time = Inf
prob = FVMProblem(mesh, BCs;
    diffusion_function,
    diffusion_parameters,
    reaction_function=reaction,
    final_time=final_time,
    initial_condition=initial_guess,
    steady=true)
sol = solve(prob, DynamicSS(TRBDF2(linsolve=KrylovJL_GMRES())))
fig = Figure(fontsize=38)
ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"y", width=600, height=300)
pt_mat = Matrix(get_points(tri)')
T_mat = [T[j] for T in each_triangle(tri), j in 1:3]
msh = mesh!(ax, pt_mat, T_mat, color=sol, colorrange=(0, 900))
Colorbar(fig[1, 2], msh)
resize_to_layout!(fig)
exact = [(R₂^2 - norm(p)^2) / (4D) + R₁^2 * log(norm(p) / R₂) / (2D) for p in each_point(tri)]
error_fnc = (fvm_sol, exact_sol) -> 100abs(fvm_sol - exact_sol) / maximum(exact)
@test median(error_fnc.(exact, sol)) < 0.1
```

![Annulus mean exit time](https://github.com/DanielVandH/FiniteVolumeMethod.jl/blob/main/test/figures/annulus_mean_exit_time.png?raw=true)

Now let's take a perturbed annulus. In particular, we let the inner boundary be $\mathcal R_1(\theta) = R_1(1 + \varepsilon g_1(\theta))$ with $g_1(\theta) = \sin(3\theta) + \cos(5\theta)$ and $\varepsilon = 1/20$, and the outer boundary is $\mathcal R_2(\theta) = R_2(1 + \varepsilon g_2(\theta))$ with $g_2(\theta) = \cos(3\theta)$.

```julia
using FiniteVolumeMethod
using DelaunayTriangulation
using CairoMakie
using OrdinaryDiffEq
using SteadyStateDiffEq
using Test
using LinearSolve
using Krylov

R₁ = 2.0
R₂ = 3.0
g₁ = θ -> sin(3θ) + cos(5θ)
g₂ = θ -> cos(3θ)
θ = LinRange(0, 2π, 250)
ε = 1/20
inner_R = R₁ .* (1 .+ ε .* g₁.(θ))
outer_R = R₂ .* ( 1 .+ ε .* g₂.(θ))
x = [
    [outer_R .* cos.(θ)],
    [reverse(inner_R .* cos.(θ))] # inner boundaries are clockwise
]
y = [
    [outer_R .* sin.(θ)],
    [reverse(inner_R .* sin.(θ))] # inner boundaries are clockwise
]
tri = generate_mesh(x, y, 0.1; gmsh_path=GMSH_PATH)
mesh = FVMGeometry(tri)
outer_bc = (x, y, t, u, p) -> zero(u)
inner_bc = (x, y, t, u, p) -> zero(u)
type = [:D, :N]
BCs = BoundaryConditions(mesh, [outer_bc, inner_bc], type)
D = 6.25e-4
initial_guess = zeros(num_points(tri))
diffusion_function = (x, y, t, u, D) -> D
diffusion_parameters = D
reaction = (x, y, t, u, p) -> one(u)
final_time = Inf
prob = FVMProblem(mesh, BCs;
    diffusion_function,
    diffusion_parameters,
    reaction_function=reaction,
    final_time=final_time,
    initial_condition=initial_guess,
    steady=true)
sol = solve(prob, DynamicSS(TRBDF2(linsolve=KrylovJL_GMRES())))
fig = Figure(fontsize=38)
ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"y", width=600, height=300)
pt_mat = Matrix(get_points(tri)')
T_mat = [T[j] for T in each_triangle(tri), j in 1:3]
msh = mesh!(ax, pt_mat, T_mat, color=sol, colorrange=(0, 900))
Colorbar(fig[1, 2], msh)
resize_to_layout!(fig)
```

![Perturbed annulus mean exit time](https://github.com/DanielVandH/FiniteVolumeMethod.jl/blob/main/test/figures/annulus_mean_exit_time.png?raw=true)