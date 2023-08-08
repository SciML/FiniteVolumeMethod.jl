# Example II: Diffusion equation in a wedge with mixed boundary conditions 

Now we consider the following problem defined on a wedge with angle $\alpha$ and mixed boundary conditions:

```math
\begin{equation*}
\begin{array}{rcll}
\dfrac{\partial u(r, \theta, t)}{\partial t} & = & \boldsymbol{\nabla}^2 u(r, \theta, t), & 0 < r < 1, 0 < \theta < \alpha, t>0, \\
\dfrac{\partial u(r, 0, t)}{\partial \theta} & = & 0, & 0 < r < 1, t>0, \\
\dfrac{\partial u(r, \alpha,t)}{\partial \theta} & = & 0, & 0 < \theta < \alpha, t>0, \\
u(r,\theta,0) & = & f(r, \theta), & 0 < r < 1, 0< \theta < \alpha.
\end{array}
\end{equation*}
```

(The exact solution to this problem, found by writing $u(r, \theta, t) = \mathrm{e}^{-\lambda t}v(r, \theta)$ and then using separation of variables, can be shown to take the form

```math
u(r, \theta, t) = \frac12\sum_{m=1}^\infty A_{0,m}\mathrm{e}^{-\zeta_{0,m}^2t}J_0\left(\zeta_{0,m}r\right) + \sum_{n=1}^\infty\sum_{m=1}^\infty A_{n,m}\mathrm{e}^{-\zeta_{n,m}^2t}J_{n\mathrm{\pi}/\alpha}\left(\zeta_{n\mathrm{\pi}/\alpha, m}r\right)\cos\left(\frac{n\mathrm{\pi}\theta}{\alpha}\right),
```math

where, assuming $f$ can be expanded into a Fourier-Bessel series,

```math
A_{n, m} = \frac{4}{\alpha J_{n\mathrm{\pi}/\alpha + 1}^2\left(\zeta_{n\mathrm{\pi}/\alpha,m}\right)}\int_0^1\int_0^\alpha f(r, \theta)J_{n\mathrm{\pi}/\alpha}\left(\zeta_{n\mathrm{\pi}/\alpha,m}r\right)\cos\left(\frac{n\mathrm{\pi}\theta}{\alpha}\right)r~\mathrm{d}r~\mathrm{d}\theta, \quad n=0,1,2,\ldots,m=1,2,\ldots,
```

and we write the roots of $J_\mu$, the $\zeta_{\mu, m}$ such that $J_\mu(\zeta_{\mu, m}) = 0$, in the form $0 < \zeta_{\mu, 1} < \zeta_{\mu, 2} < \cdots$ with $\zeta_{\mu, m} \to \infty$ as $m \to \infty$. This is the exact solution we compare to in the tests; comparisons not shown here.) We take $\alpha = \mathrm{\pi}/4$ and $f(r, \theta) = 1 - r$. 

Note that the PDE is provided in polar form, but Cartesian coordinates are assumed for the operators in our code. The conversion is easy, noting that the two Neumann conditions are just equations of the form $\boldsymbol{\nabla} u \boldsymbol{\cdot} \hat{\boldsymbol{n}} = 0$. Moreover, although the right-hand side of the PDE is given as a Laplacian, recall that $\boldsymbol{\nabla}^2 = \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{\nabla}$, so we can write $\partial u/\partial t + \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{q} = 0$, where $\boldsymbol{q} = -\boldsymbol{\nabla} u$, or `q(x, y, t, α, β, γ, p) = (-α, -β)` in the notation in our code.

## Setting up the problem 

Let us now solve the problem. Again, we start by defining the mesh. Since the boundary condition is different on each segment, we keep each segment as a different vector.
```julia
using DelaunayTriangulation, FiniteVolumeMethod, ElasticArrays

## Step 1: Generate the mesh 
n = 50
α = π / 4

# The bottom edge 
x₁ = [0.0,1.0]
y₁ = [0.0,0.0]

# Arc 
r₂ = LinRange(1, 1, n)
θ₂ = LinRange(0, α, n)
x₂ = @. r₂ * cos(θ₂)
y₂ = @. r₂ * sin(θ₂)

# Upper edge 
x₃ = [cos(α), 0.0]
y₃ = [sin(α), 0.0]

# Combine and create the mesh 
x = [x₁, x₂, x₃]
y = [y₁, y₂, y₃]
boundary_nodes, points = convert_boundary_points_to_indices(x, y; existing_points = ElasticMatrix{Float64}(undef, 2, 0))
tri = triangulate(points; boundary_node)
A = get_total_area(tri)
refine!(tri; max_area = 1e-4A)
mesh = FVMGeometry(tri)
```

Now we define the boundary conditions.
```julia
lower_bc = ((x, y, t, u::T, p) where {T}) -> zero(T)
arc_bc = ((x, y, t, u::T, p) where {T}) -> zero(T)
upper_bc = ((x, y, t, u::T, p) where {T}) -> zero(T)
types = (:N, :D, :N)
boundary_functions = (lower_bc, arc_bc, upper_bc)
BCs = BoundaryConditions(mesh, boundary_functions, types)
```

Next, the PDE itself:
```julia
f = (x, y) -> 1 - sqrt(x^2 + y^2)
D = ((x, y, t, u::T, p) where {T}) -> one(T)
points = get_points(tri)
u₀ = f.(points[1, :], points[2, :])
final_time = 0.1 # Do not need iip_flux = true or R(x, y, t, u, p) = 0, these are defaults 
prob = FVMProblem(mesh, BCs; diffusion_function=D, initial_condition=u₀, final_time)
```
This formulation uses the diffusion function rather than the flux function, but you could also use the flux function formulation:
```julia 
flux = (q, x, y, t, α, β, γ, p) -> (q[1] = -α; q[2] = -β; nothing)
prob2 = FVMProblem(mesh, BCs; diffusion_function=D, initial_condition=u₀, final_time)
```

## Solving the problem and visualising the solution 

Finally, we can solve and visualise the problem. 
```julia 
using OrdinaryDiffEq, LinearSolve

alg = Rosenbrock23(linsolve=UMFPACKFactorization())
sol = solve(prob, alg; saveat=0.025)
```

```julia
using CairoMakie 
fig = Figure(resolution=(2068.72f0, 686.64f0), fontsize=38)
for (i, j) in zip(1:3, (1, 3, 5))
    ax = Axis(fig[1, i], width=600, height=600)
    tricontourf!(ax, tri, sol.u[j], levels=0:0.01:1, colormap=:matter)
    tightlimits!(ax)
end
fig
```

```@raw html
<figure>
    <img src='../figures/diffusion_equation_wedge_test.png', alt='Diffusion equation on a wedge'><br>
</figure>
```

Here is a more detailed comparison with the exact solution.

```@raw html
<figure>
    <img src='../figures/heat_equation_wedge_test_error.png', alt='Diffusion equation on a wedge compared to exact solution'><br>
</figure>
```