# Example I: Diffusion Equation on a Square Plate

Please ensure you have read the List of Examples section before proceeding.

We first consider the problem of diffusion on a square plate,

```math
\begin{equation*}
\begin{array}{rcll}
\displaystyle
\frac{\partial u(x, y, t)}{\partial t} &=& \dfrac19\boldsymbol{\nabla}^2 u(x, y, t) & (x, y) \in \Omega,t>0, \\
u(x, y, t) &= & 0 & (x, y) \in \partial \Omega,t>0, \\
u(x, y, 0) &= & f(x, y) &(x,y)\in\Omega,
\end{array}
\end{equation*}
```

where $\Omega = [0, 2]^2$ and $f(x, y) = 50$ if $y \leq 1$ and $f(x, y) = 0$ if $y>1$. (This problem has an exact solution 

```math
u(x, y, t) = \dfrac{200}{\mathrm{\pi}^2}\sum_{m=1}^\infty\sum_{n=1}^\infty \frac{\left[1+(-1)^{m+1}\right]\left[1-\cos\left(\frac{n\mathrm{\pi}}{2}\right)\right]}{mn}\sin\left(\frac{m\mathrm{\pi}x}{2}\right)\sin\left(\frac{n\mathrm{\pi}y}{2}\right)\mathrm{e}^{-\frac{1}{36}\mathrm{\pi}^2(m^2+n^2)t},
```

and we compare our results to this exact solution in the tests. See e.g. [here](http://ramanujan.math.trinity.edu/rdaileda/teach/s12/m3357/lectures/lecture_3_6_short.pdf) for a derivation of the exact solution. Comparisons not shown here.)

## Setting up the problem 

The first step is to define the mesh:
```julia
using FiniteVolumeMethod, DelaunayTriangulation
a, b, c, d = 0.0, 2.0, 0.0, 2.0
p1 = (a, c)
p2 = (b, c)
p3 = (b, d)
p4 = (a, d)
points = [p1, p2, p3, p4]
boundary_nodes = [1, 2, 3, 4, 1]
tri = triangulate(points; boundary_nodes)
A = get_total_area(tri)
refine!(tri; max_area=1e-4A)
mesh = FVMGeometry(tri)
```

Here I start by defining the triangulation of the boundary nodes, which I then refine to get a dense mesh. The representation of the mesh for the finite volume method is then created with `FVMGeometry`.

Now having defined the mesh, let us define the boundary conditions. We have a homogeneous Dirichlet condition, so let us simply set
```julia
bc = ((x, y, t, u::T, p) where {T}) -> zero(T)
type = :Dirichlet # or :D or :dirichlet or "D" or "Dirichlet"
BCs = BoundaryConditions(mesh, bc, type)
```

Next we must define the actual PDE. The initial condition, diffusion, and reaction functions are defined as follows:
```julia
f = (x, y) -> y ≤ 1.0 ? 50.0 : 0.0 # initial condition 
D = (x, y, t, u, p) -> 1 / 9 # You could also define flux = (q, x, y, t, α, β, γ, p) -> (q[1] = -α/9; q[2] = -β/9)
R = ((x, y, t, u::T, p) where {T}) -> zero(T)
```
Using `f`, we compute the initial condition vector:
```julia
points = get_points(tri)
u₀ = @views f.(first.(points), last.(points))
```
We want the flux function to be computed in-place when it is constructed from `D`, so we will set `iip_flux = true`. Lastly, we want to solve up to `t = 0.5`, so `final_time = 0.5` (`initial_time = 0.0` is the default for the initial time). 
```julia
iip_flux = true
final_time = 0.5
prob = FVMProblem(mesh, BCs; iip_flux,
    diffusion_function=D, reaction_function=R,
    initial_condition=u₀, final_time)
```
This now defines our problem. Note that the delay function has been defined as the identity function, and the flux function has been computed from the diffusion function so that $\boldsymbol{q}(x, y, t, \alpha, \beta, \gamma) = (-\alpha/9,-\beta/9)^{\mathsf T}$.

## Solving the problem 

Now having the problem, we can solve it:
```julia
using OrdinaryDiffEq, LinearSolve

alg = TRBDF2(linsolve=KLUFactorization(), autodiff=true)
sol = solve(prob, alg; specialization=SciMLBase.FullSpecialize, saveat=0.05)
```
```julia
julia> sol
retcode: Success
Interpolation: 1st order linear
t: 11-element Vector{Float64}:
 0.0
 0.05
 0.1
 ⋮
 0.45
 0.5
u: 11-element Vector{Vector{Float64}}:
 [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0  …  0.0, 50.0, 50.0, 50.0, 0.0, 50.0, 50.0, 0.0, 0.0, 50.0]
 [...] (truncated)
```

## Visualising the solution 

You can use `sol` as you would any other solution from `DifferentialEquations.jl` (e.g. `sol(t)` returns the solution at time `t`). To visualise the solution at the times `t = 0.0`, `t = 0.25`, and `t = 0.5`, the following code can be used:
```julia
using CairoMakie
T_mat = [T[j] for T in each_triangle(tri), j in 1:3]
fig = Figure(resolution=(2068.72f0, 686.64f0), fontsize=38)
ax = Axis(fig[1, 1], width=600, height=600)
xlims!(ax, a, b)
ylims!(ax, c, d)
mesh!(ax, points, T_mat, color=sol.u[1], colorrange=(0, 50), colormap=:matter)
ax = Axis(fig[1, 2], width=600, height=600)
xlims!(ax, a, b)
ylims!(ax, c, d)
mesh!(ax, points, T_mat, color=sol.u[6], colorrange=(0, 50), colormap=:matter)
ax = Axis(fig[1, 3], width=600, height=600)
xlims!(ax, a, b)
ylims!(ax, c, d)
mesh!(ax, points, T_mat, color=sol.u[11], colorrange=(0, 50), colormap=:matter)
```

```@raw html
<figure>
    <img src='../figures/heat_equation_test.png', alt='Diffusion equation on a square plate'><br>
</figure>
```

Here is a comparison with the exact solution.

```@raw html
<figure>
    <img src='../figures/heat_equation_test_error.png', alt='Diffusion equation on a square plate comparison'><br>
</figure>
```