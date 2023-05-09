# Example III: Reaction-diffusion equation with a time-dependent Dirichlet boundary condition on a disk 

Please ensure you have read the List of Examples section before proceeding.

Now we consider

```math
\begin{equation*}
\begin{array}{rcll}
\dfrac{\partial u(r, \theta, t)}{\partial t} & = & \boldsymbol{\nabla} \boldsymbol{\cdot} [u\boldsymbol{\nabla} u] + u(1-u), & 0 < r < 1, 0 < \theta < 2\mathrm{\pi}, \\
\dfrac{\mathrm{d}u(1, \theta, t)}{\mathrm{d}t} & = & u(1, \theta, t), & 0 < \theta < 2\mathrm{\pi}, t > 0,  \\
u(r, \theta, 0) & = & \sqrt{I_0(\sqrt{2}r)},
\end{array}
\end{equation*}
```

where $I_0$ is the modified Bessel function of the first kind of order zero. (The solution to this problem is $u(r, \theta, t) = \mathrm{e}^t\sqrt{I_0(\sqrt{2}r)}$ (see [Bokhari et al. (2008)](https://doi.org/10.1016/j.na.2007.11.012)) This is what we compare to in the tests, and again these comparisons are not shown here.) In this case, the diffusion function is $D(x, y, t, u) = u$ and the reaction function is $R(x, y, t, u) = u(1-u)$, or equivalently the flux function is 

```math
\boldsymbol{q}(x, y, t, \alpha, \beta, \gamma) = \left(-\alpha\left(\alpha x + \beta y + \gamma\right), -\beta\left(\alpha x + \beta y + \gamma\right)\right)^{\mathsf T}. 
```

The following code solves this problem numerically.
```julia 
using FiniteVolumeMethod, DelaunayTriangulation, ElasticArrays

## Step 1: Generate the mesh 
r = LinRange(1, 1, 100)
θ = LinRange(0, 2π, 100)
x = @. r * cos(θ)
y = @. r * sin(θ)
x[end] = x[begin]
y[end] = y[begin] # make sure the curve connects at the endpoints
boundary_nodes, points = convert_boundary_points_to_indices(x, y; existing_points=ElasticMatrix{Float64}(undef, 2, 0))
tri = triangulate(points; boundary_nodes)
A = get_total_area(tri)
refine!(tri; max_area=1e-4A)
mesh = FVMGeometry(tri)
points = get_points(tri)

## Step 2: Define the boundary conditions 
bc = (x, y, t, u, p) -> u
types = :dudt
BCs = BoundaryConditions(mesh, bc, types)

## Step 3: Define the actual PDE  
using Bessels

f = (x, y) -> sqrt(besseli(0.0, sqrt(2) * sqrt(x^2 + y^2)))
D = (x, y, t, u, p) -> u
R = (x, y, t, u, p) -> u * (1 - u)
u₀ = [f(points[:, i]...) for i in axes(points, 2)]
final_time = 0.10
prob = FVMProblem(mesh, BCs; diffusion_function=D, reaction_function=R, initial_condition=u₀, final_time)

## Step 4: Solve
using OrdinaryDiffEq, LinearSolve

alg = FBDF(linsolve=UMFPACKFactorization())
sol = solve(prob, alg; saveat=0.025)

## Step 5: Visualisation 
using CairoMakie 

pt_mat = Matrix(points')
T_mat = [T[j] for T in each_triangle(tri), j in 1:3]
fig = Figure(resolution=(2131.8438f0, 684.27f0), fontsize=38)
ax = Axis(fig[1, 1], width=600, height=600)
mesh!(ax, pt_mat, T_mat, color=sol.u[1], colorrange=(1, 1.1), colormap=:matter)
ax = Axis(fig[1, 2], width=600, height=600)
mesh!(ax, pt_mat, T_mat, color=sol.u[3], colorrange=(1, 1.1), colormap=:matter)
ax = Axis(fig[1, 3], width=600, height=600)
mesh!(ax, pt_mat, T_mat, color=sol.u[5], colorrange=(1, 1.1), colormap=:matter)
```

```@raw html
<figure>
    <img src='../figures/reaction_diffusion_test.png', alt='Reaction-diffusion equation on a circle solution'><br>
</figure>
```

Here is a more detailed comparison with the exact solution.

```@raw html
<figure>
    <img src='../figures/reaction_diffusion_equation_test_error.png', alt='Reaction-diffusion equation on a circle solution compared to the exact solution'><br>
</figure>
```