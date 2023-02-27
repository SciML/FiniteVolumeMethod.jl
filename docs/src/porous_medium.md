# Example IV: Porous-medium equation 

Please ensure you have read the List of Examples section before proceeding.

## No source 

We now consider the Porous medium equation,

```math
\dfrac{\partial u}{\partial t} = D\boldsymbol{\nabla} \boldsymbol{\cdot} \left[u^{m-1} \boldsymbol{\nabla}u\right],
```

with initial condition $u(x, y, 0) = M\delta(x, y)$ where $\delta(x, y)$ is the Dirac delta function and $M = \iint_{\mathbb R^2} u(x, y, t)~\mathrm{d}A$. The diffusion function here is $D(x, y, t, u) = Du^{m-1}$. We approximate $\delta(x, y)$ by 

```math
\delta(x, y) \approx g(x, y) = \frac{1}{\varepsilon^2 \mathrm{\pi}}\exp\left[-\frac{1}{\varepsilon^2}\left(x^2 + y^2\right)\right],
```

taking $\varepsilon = 0.1$. This equation has an exact solution (see e.g. Section 17.5 of the *The porous medium equation: Mathematical theory* by J. L. Vázquez (2007)) 

```math
u(x, y, t) = \begin{cases} (Dt)^{-1/m}\left[\left(\dfrac{M}{4\mathrm{\pi}}\right)^{(m-1)/m} - \dfrac{m-1}{4m}\left(x^2+y^2\right)(Dt)^{-1/m}\right]^{1/(m-1)} & x^2 + y^2 < R_{m, M}(Dt)^{1/m}, \\
0 & x^2 + y^2 \geq R_{m, M}(Dt)^{1/m},\end{cases}
```

where $R_{m, M} = [4m/(m-1)][M/(4\mathrm{\pi})]^{(m-1)/m}$. This equation has compact support, so we replace $\mathbb R^2$ by the domain $\Omega = [-R_{m, M}^{1/2}(DT)^{1/2m}, R_{m, M}^{1/2}(DT)^{1/2m}]^2$, where $T$ is the time that we solve up to, and we take Dirichlet boundary conditions on $\partial\Omega$. We solve this problem as follows, taking $m = 2$, $M = 0.37$, $D = 2.53$, and $T = 12$. Note the use of the parameters.
```julia
using DelaunayTriangulation, FiniteVolumeMethod

## Step 0: Define all the parameters 
m = 2
M = 0.37
D = 2.53
final_time = 12.0
ε = 0.1

## Step 1: Define the mesh 
RmM = 4m / (m - 1) * (M / (4π))^((m - 1) / m)
L = sqrt(RmM) * (D * final_time)^(1 / (2m))
r = 0.1
tri = generate_mesh(-L, L, -L, L, r; gmsh_path=GMSH_PATH)
mesh = FVMGeometry(tri)
points = get_points(tri)

## Step 2: Define the boundary conditions 
bc = ((x, y, t, u::T, p) where {T}) -> zero(T)
types = :D
BCs = BoundaryConditions(mesh, bc, types)

## Step 3: Define the actual PDE  
f = (x, y) -> M * 1 / (ε^2 * π) * exp(-1 / (ε^2) * (x^2 + y^2))
diff_fnc = (x, y, t, u, p) -> p[1] * u^(p[2] - 1)
diff_parameters = (D, m)
u₀ = [f(points[:, i]...) for i in axes(points, 2)]
prob = FVMProblem(mesh, BCs; diffusion_function=diff_fnc,
    diffusion_parameters=diff_parameters, initial_condition=u₀, final_time)

## Step 4: Solve
using LinearSolve, OrdinaryDiffEq

alg = TRBDF2(linsolve=KLUFactorization())
sol = solve(prob, alg; saveat=3.0)

## Step 5: Visualisation
using CairoMakie

pt_mat = Matrix(points')
T_mat = [T[j] for T in each_triangle(tri), j in 1:3]
fig = Figure(resolution=(2131.8438f0, 684.27f0), fontsize=38)
ax = Axis(fig[1, 1], width=600, height=600)
mesh!(ax, pt_mat, T_mat, color=sol.u[1], colorrange=(0.0, 0.05), colormap=:matter)
ax = Axis(fig[1, 2], width=600, height=600)
mesh!(ax, pt_mat, T_mat, color=sol.u[3], colorrange=(0.0, 0.05), colormap=:matter)
ax = Axis(fig[1, 3], width=600, height=600)
mesh!(ax, pt_mat, T_mat, color=sol.u[5], colorrange=(0.0, 0.05), colormap=:matter)
```
![Porous-medium equation with m=2](https://github.com/DanielVandH/FiniteVolumeMethod.jl/blob/main/test/figures/porous_medium_test.png?raw=true)

## Linear source

We can continue this example with the Porous medium equation by considering the same equation except with a linear source:

```math
\dfrac{\partial u}{\partial t} = D\boldsymbol{\nabla} \boldsymbol{\cdot} \left[u^{m-1}\boldsymbol{\nabla} u\right] + \lambda u, \quad \lambda>0. 
```

This equation has an exact solution given by 

```math
u(x, y, t) = \mathrm{e}^{\lambda t}v\left(x, y, \frac{D}{\lambda(m-1)}\left[\mathrm{e}^{\lambda(m-1)t} - 1\right]\right),
```

where $u(x, y, 0) = M\delta(x, y)$ and $v$ is the exact solution we gave above except with $D=1$. This is what we use for assessing the solution in the tests - not shown here. The domain we use is now $\Omega = [-R_{m, M}^{1/2}\tau(T)^{1/2m}, R_{m,M}^{1/2}\tau(T)^{1/2m}]^2$, where $\tau(T) = \frac{D}{\lambda(m-1)}[\mathrm{e}^{\lambda(m-1)T}-1]$. The code below solves this problem.

```julia
using OrdinaryDiffEq, LinearSolve, FiniteVolumeMethod, DelaunayTriangulation, CairoMakie

## Step 0: Define all the parameters 
m = 3.4
M = 2.3
D = 0.581
λ = 0.2
final_time = 10.0
ε = 0.1

## Step 1: Define the mesh 
RmM = 4m / (m - 1) * (M / (4π))^((m - 1) / m)
L = sqrt(RmM) * (D / (λ * (m - 1)) * (exp(λ * (m - 1) * final_time) - 1))^(1 / (2m))
r = 0.07
tri = generate_mesh(-L, L, -L, L, r; gmsh_path=GMSH_PATH)
mesh = FVMGeometry(tri)
points = get_points(tri)

## Step 2: Define the boundary conditions 
bc = ((x, y, t, u::T, p) where {T}) -> zero(T)
types = :D
BCs = BoundaryConditions(mesh, bc, types)

## Step 3: Define the actual PDE  
f = (x, y) -> M * 1 / (ε^2 * π) * exp(-1 / (ε^2) * (x^2 + y^2))
diff_fnc = (x, y, t, u, p) -> p[1] * abs(u)^(p[2] - 1)
reac_fnc = (x, y, t, u, p) -> p[1] * u
diff_parameters = (D, m)
react_parameter = λ
u₀ = [f(points[:, i]...) for i in axes(points, 2)]
prob = FVMProblem(mesh, BCs; diffusion_function=diff_fnc,
    diffusion_parameters=diff_parameters,
    reaction_function=reac_fnc, reaction_parameters=react_parameter,
    initial_condition=u₀, final_time)

## Step 4: Solve
alg = TRBDF2(linsolve=KLUFactorization())
sol = solve(prob, alg; saveat=2.5)

## Step 5: Visualisation 
pt_mat = Matrix(points')
T_mat = [T[j] for T in each_triangle(tri), j in 1:3]
fig = Figure(resolution=(2131.8438f0, 684.27f0), fontsize=38)
ax = Axis(fig[1, 1], width=600, height=600)
mesh!(ax, pt_mat, T_mat, color=sol.u[1], colorrange=(0.0, 0.5), colormap=:matter)
ax = Axis(fig[1, 2], width=600, height=600)
mesh!(ax, pt_mat, T_mat, color=sol.u[3], colorrange=(0.0, 0.5), colormap=:matter)
ax = Axis(fig[1, 3], width=600, height=600)
mesh!(ax, pt_mat, T_mat, color=sol.u[5], colorrange=(0.0, 0.5), colormap=:matter)
```
![Porous-medium equation with linear source](https://github.com/DanielVandH/FiniteVolumeMethod.jl/blob/main/test/figures/porous_medium_linear_source_test.png?raw=true)