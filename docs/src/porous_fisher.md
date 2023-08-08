# Example V: Porous-Fisher equation and travelling waves

Please ensure you have read the List of Examples section before proceeding.

We now consider a more involved example, where we discuss the travelling wave solutions of the Porous-Fisher equation and discuss how we test a more complicated problem. We consider:

```math
\begin{equation*}
\begin{array}{rcll}
\dfrac{\partial u(x, y, t)}{\partial t} &=& D\boldsymbol{\nabla} \boldsymbol{\cdot} [u \boldsymbol{\nabla} u] + \lambda u(1-u), & 0 < x < a, 0 < y < b, t > 0, \\
u(x, 0, t) & = & 1 & 0 < x < a, t > 0, \\
u(x, b, t) & = & 0 & 0 < x < a , t > 0, \\
\dfrac{\partial u(0, y, t)}{\partial x} & = & 0 & 0 < y < b, t > 0, \\
\dfrac{\partial u(a, y, t)}{\partial x} & = & 0 & 0 < y < b, t > 0, \\
u(x, y, 0) & = & f(y) & 0 \leq x \leq a, 0 \leq y \leq b. 
\end{array}
\end{equation*}
```

This problem is defined on the rectangle $[0, a] \times [0, b]$, and we assume that $b \gg a$ so that the rectangle is much taller than it is wide. This problem has $u$ ranging from $u=1$ at the bottom of the rectangle down to $u=0$ at the top of the rectangle, with no flux conditions on the two vertical walls. The function $f(y)$ is taken to be independent of $x$. This setup implies that the solution along each constant line $x = x_0$ should be about the same, i.e. the problem is invariant in $x$. If indeed we have $u(x, y, t) \equiv u(y, t)$, then the PDE becomes

```math
\dfrac{\partial u(y, t)}{\partial t} = D\dfrac{\partial}{\partial y}\left(u\dfrac{\partial u}{\partial y}\right) + \lambda u(1-u),
```

which has travelling wave solutions. Following the analysis from Section 13.4 of *Mathematical biology I: An introduction* by J. D. Murray (2002), we can show that a travelling wave solution to the one-dimensional problem above is

```math
u(y, t) = \left(1 - \mathrm{e}^{c_{\min}z}\right)\left[z \leq 0\right]
```

where $c_{\min} = \sqrt{\lambda/(2D)}$, $c = \sqrt{D\lambda/2}$, and $z = x - ct$ is the travelling wave coordinate. This travelling wave would match our problem exactly if the rectangle were instead $[0, a] \times \mathbb R$, but by choosing $b$ large enough we can at least emulate the travelling wave behaviour closely; homogeneous Neumann conditions are to ensure no energy is lost, thus allowing the travelling waves to exist. Note also that the approximations of the solution with $u(y, t)$ above will only be accurate for large time.

Let us now solve the problem. For this problem, rather than using `generate_mesh` we will use a structured triangulation with `triangulate_structured`. This will make it easier to test the $x$ invariance.

```julia
using DelaunayTriangulation, FiniteVolumeMethod

## Step 1: Define the mesh 
a, b, c, d, Nx, Ny = 0.0, 3.0, 0.0, 40.0, 60, 80
tri = triangulate_rectangle(a, b, c, d, Nx, Ny; single_boundary = false)
mesh = FVMGeometry(tri)
points = get_points(tri)

## Step 2: Define the boundary conditions 
a₁ = ((x, y, t, u::T, p) where {T}) -> one(T)
a₂ = ((x, y, t, u::T, p) where {T}) -> zero(T)
a₃ = ((x, y, t, u::T, p) where {T}) -> zero(T)
a₄ = ((x, y, t, u::T, p) where {T}) -> zero(T)
bc_fncs = (a₁, a₂, a₃, a₄)
types = (:D, :N, :D, :N)
BCs = BoundaryConditions(mesh, bc_fncs, types)

## Step 3: Define the actual PDE  
f = ((x::T, y::T) where {T}) -> zero(T)
diff_fnc = (x, y, t, u, p) -> p * u
reac_fnc = (x, y, t, u, p) -> p * u * (1 - u)
D, λ = 0.9, 0.99
diff_parameters = D
reac_parameters = λ
final_time = 50.0
u₀ = [f(x, y) for (x, y) in points]
prob = FVMProblem(mesh, BCs; diffusion_function=diff_fnc, reaction_function=reac_fnc,
    diffusion_parameters=diff_parameters, reaction_parameters=reac_parameters,
    initial_condition=u₀, final_time)

## Step 4: Solve
using LinearSolve, OrdinaryDiffEq

alg = TRBDF2(linsolve=KLUFactorization())
sol = solve(prob, alg; saveat=0.5)
```

This gives us our solution. To verify the $x$-invariance, something like the following suffices:
```julia
using Test, StatsBase

u_mat = [reshape(u, (Nx, Ny)) for u in sol.u]
all_errs = zeros(length(sol))
err_cache = zeros((Nx - 1) * Ny)
for i in eachindex(sol)
    u = u_mat[i]
    ctr = 1
    for j in union(1:((Nx÷2)-1), ((Nx÷2)+1):Nx)
        for k in 1:Ny
            err_cache[ctr] = 100abs(u[j, k] .- u[Nx÷2, k])
            ctr += 1
        end
    end
    all_errs[i] = mean(err_cache)
end
@test all(all_errs .< 0.05)
```
In this code, we test the $x$-invariance by seeing if the $u(x, y) \approx u(x_0, y)$ for each $x$, where $x_0$ is the midpoint $a/2$.

To now see the travelling wave behaviour, we use the following:
```julia 
large_time_idx = findfirst(sol.t .== 10)
c = sqrt(λ / (2D))
cₘᵢₙ = sqrt(λ * D / 2)
zᶜ = 0.0
exact_solution = ((z::T) where {T}) -> ifelse(z ≤ zᶜ, 1 - exp(cₘᵢₙ * (z - zᶜ)), zero(T))
travelling_wave_values = zeros(Ny, length(sol) - large_time_idx + 1)
z_vals = zeros(Ny, length(sol) - large_time_idx + 1)
for (i, t_idx) in pairs(large_time_idx:lastindex(sol))
    u = u_mat[t_idx]
    τ = sol.t[t_idx]
    for k in 1:Ny
        y = c + (k - 1) * (d - c) / (Ny - 1)
        z = y - c * τ
        z_vals[k, i] = z
        travelling_wave_values[k, i] = u[Nx÷2, k]
    end
end
exact_z_vals = collect(LinRange(extrema(z_vals)..., 500))
exact_travelling_wave_values = exact_solution.(exact_z_vals)
```
The results we obtain are shown below, with the exact travelling wave from the one-dimensional problem shown in red in the fourth plot and the numerical solutions are the other curves. 

```julia 
using CairoMakie 

# The solution 
fig = Figure(resolution=(3024.72f0, 686.64f0), fontsize=38)
for (i, j) in zip(1:3, (1, 51, 101))
    ax = Axis(fig[1, i], width=600, height=600)
    tricontourf!(ax, tri, sol.u[j], levels=0:0.05:1, colormap=:matter)
    tightlimits!(ax)
end
fig

# The wave comparisons 
ax = Axis(fig[1, 4], width=900, height=600)
colors = cgrad(:matter, length(sol) - large_time_idx + 1; categorical=false)
[lines!(ax, z_vals[:, i], travelling_wave_values[:, i], color=colors[i], linewidth=2) for i in 1:(length(sol)-large_time_idx+1)]
lines!(ax, exact_z_vals, exact_travelling_wave_values, color=:red, linewidth=4, linestyle=:dash)
```

```@raw html
<figure>
    <img src='../figures/travelling_wave_problem_test.png', alt='Travelling wave problem'><br>
</figure>
```