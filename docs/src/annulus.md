# Example VII: Diffusion equation on an annulus

Please ensure you have read the List of Examples section before proceeding.

We now consider another diffusion problem, except now we demonstrate how we can solve a problem on a multiply-connected domain. The problem we consider comes from [http://onelab.info/wiki/Tutorial/Heat_equation_with_Dirichlet_boundary_control](http://onelab.info/wiki/Tutorial/Heat_equation_with_Dirichlet_boundary_control), namely

```math
\begin{equation*}
\begin{array}{rcll}
\dfrac{\partial u}{\partial t} & = & \boldsymbol{\nabla}^2 u & \boldsymbol{x} \in \Omega, t>0, \\
\boldsymbol{\nabla} u \boldsymbol{\cdot}  \hat{\boldsymbol n} &=& 0 & \boldsymbol{x} \in \mathcal D(0, 1), t>0, \\
u(x, y, t) & = & c(t) & \boldsymbol{x} \in \mathcal D(0, 0.2),  t>0, \\
u(x, y, 0) & = & u_0(x, y) & \boldsymbol{x} \in \Omega,
\end{array}
\end{equation*}
```

where $\mathcal D(0, r)$ is a circle of radius $r$ centred at the origin, $\Omega$ is the annulus bteween $\mathcal D(0, 0.2)$ and $\mathcal D(0, 1)$, $c(t) = 50[1-\exp(-0.5t)]$, and 

```math 
u_0(x) =10\mathrm{e}^{-25\left(\left(x + \frac12\right)^2 + \left(y + \frac12\right)^2\right)} - 10\mathrm{e}^{-45\left(\left(x - \frac12\right)^2 + \left(y - \frac12\right)^2\right)} - 5\mathrm{e}^{-50\left(\left(x + \frac{3}{10}\right)^2 + \left(y + \frac12\right)^2\right)}.
```

To define this problem, we define the problem as we have been doing, but now we take special care to define the multiply-connected domain. In particular, we define the boundary nodes according to the specification in DelaunayTriangulation.jl (see the boundary nodes discussion [here](https://danielvandh.github.io/DelaunayTriangulation.jl/dev/boundary_handling/)). The complete code is below, where we generate the mesh, and then visualise the solution.

```julia 
## Generate the mesh. 
# When specifying multiple boundary curves, the outer boundary comes first 
# and is given in counter-clockwise order. The inner boundaries then follow. 
R₁ = 0.2
R₂ = 1.0
θ = collect(LinRange(0, 2π, 100))
θ[end] = 0.0 # get the endpoints to match
x = [
    [R₂ .* cos.(θ)],
    [reverse(R₁ .* cos.(θ))]
]
y = [
    [R₂ .* sin.(θ)],
    [reverse(R₁ .* sin.(θ))]
]
boundary_nodes, points = convert_boundary_points_to_indices(x, y)
tri = triangulate(points; boundary_nodes)
A = get_total_area(tri)
refine!(tri;max_area=1e-4A)
mesh = FVMGeometry(tri)

## Define the boundary conditions 
outer_bc = (x, y, t, u, p) -> zero(u)
inner_bc = (x, y, t, u, p) -> 50.0 * (1.0 - exp(-0.5t))
type = [:N, :D]
BCs = BoundaryConditions(mesh, [outer_bc, inner_bc], type)

## Define the problem 
initial_condition_f = (x, y) -> begin
    10 * exp(-25 * ((x + 0.5) * (x + 0.5) + (y + 0.5) * (y + 0.5))) - 5 * exp(-50 * ((x + 0.3) * (x + 0.3) + (y + 0.5) * (y + 0.5))) - 10 * exp(-45 * ((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)))
end
diffusion = (x, y, t, u, p) -> one(u)
points = get_points(tri)
u₀ = [initial_condition_f(x, y) for (x, y) in points]
final_time = 2.0
prob = FVMProblem(mesh, BCs;
    diffusion_function=diffusion,
    final_time=final_time,
    initial_condition=u₀)

## Solve the problem 
alg = TRBDF2(linsolve=KLUFactorization())
sol = solve(prob, alg, parallel=false, saveat=0.2)

## Visualise 
fig = Figure(resolution=(2068.72f0, 686.64f0), fontsize=38)
for (i, j) in zip(1:3, (1, 6, 11))
    ax = Axis(fig[1, i], width=600, height=600)
    tricontourf!(ax, tri, sol.u[j], levels=-10:2:40, colormap=:viridis)
    tightlimits!(ax)
end
fig
```

```@raw html
<figure>
    <img src='../figures/annulus_test.png', alt='Solution on an annulus'><br>
</figure>
```