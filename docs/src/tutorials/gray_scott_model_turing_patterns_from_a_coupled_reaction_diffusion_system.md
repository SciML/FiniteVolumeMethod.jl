```@meta
EditURL = "https://github.com/SciML/FiniteVolumeMethod.jl/tree/main/docs/src/literate_tutorials/gray_scott_model_turing_patterns_from_a_coupled_reaction_diffusion_system.jl"
```


# Gray-Scott Model: Turing Patterns from a Coupled Reaction-Diffusion System

In this tutorial, we explore some pattern formation from the
Gray-Scott model:
```math
\begin{equation}
\begin{aligned}
\pdv{u}{t} &= \varepsilon_1\grad^2u+b(1-u)-uv^2, \\
\pdv{v}{t} &= \varepsilon_2\grad^2 v - dv+uv^2,
\end{aligned}
\end{equation}
```
where $u$ and $v$ are the concentrations of two chemical species. The
initial conditions we use are:
```math
\begin{align*}
u(x, y, 0) &= 1 -\exp\left[-80\left(x^2 + y^2\right)\right], \\
v(x, y, 0) &= \exp\left[-80\left(x^2+y^2\right)\right].
\end{align*}
```
The domain we use is $[-1, 1]^2$, and we use
zero flux boundary conditions.

````julia
using FiniteVolumeMethod, DelaunayTriangulation
tri = triangulate_rectangle(-1, 1, -1, 1, 200, 200, single_boundary=true)
mesh = FVMGeometry(tri)
````

````
FVMGeometry with 40000 control volumes, 79202 triangles, and 119201 edges
````

````julia
bc = (x, y, t, (u, v), p) -> zero(u) * zero(v)
u_BCs = BoundaryConditions(mesh, bc, Neumann)
v_BCs = BoundaryConditions(mesh, bc, Neumann)
````

````
BoundaryConditions with 1 boundary condition with type Neumann
````

````julia
ε₁ = 0.00002
ε₂ = 0.00001
b = 0.04
d = 0.1
u_q = (x, y, t, α, β, γ, _ε₁) -> (-α[1] * _ε₁, -β[1] * _ε₁)
v_q = (x, y, t, α, β, γ, _ε₂) -> (-α[2] * _ε₂, -β[2] * _ε₂)
u_S = (x, y, t, (u, v), _b) -> _b * (1 - u) - u * v^2
v_S = (x, y, t, (u, v), _d) -> -_d * v + u * v^2
u_qp = ε₁
v_qp = ε₂
u_Sp = b
v_Sp = d
u_icf = (x, y) -> 1 - exp(-80 * (x^2 + y^2))
v_icf = (x, y) -> exp(-80 * (x^ 2 + y^2))
u_ic = [u_icf(x, y) for (x, y) in DelaunayTriangulation.each_point(tri)]
v_ic = [v_icf(x, y) for (x, y) in DelaunayTriangulation.each_point(tri)]
u_prob = FVMProblem(mesh, u_BCs;
    flux_function=u_q, flux_parameters=u_qp,
    source_function=u_S, source_parameters=u_Sp,
    initial_condition=u_ic, final_time=6000.0)
v_prob = FVMProblem(mesh, v_BCs;
    flux_function=v_q, flux_parameters=v_qp,
    source_function=v_S, source_parameters=v_Sp,
    initial_condition=v_ic, final_time=6000.0)
prob = FVMSystem(u_prob, v_prob)
````

````
FVMSystem with 2 equations and time span (0.0, 6000.0)
````

Now that we have our system, we can solve.

````julia
using OrdinaryDiffEq, Sundials
sol = solve(prob, CVODE_BDF(linear_solver=:GMRES), saveat=10.0, parallel=Val(false))
````

````
retcode: Success
Interpolation: 1st order linear
t: 601-element Vector{Float64}:
    0.0
   10.0
   20.0
   30.0
   40.0
    ⋮
 5960.0
 5970.0
 5980.0
 5990.0
 6000.0
u: 601-element Vector{Matrix{Float64}}:
 [1.0 1.0 … 1.0 1.0; 3.257488532207521e-70 1.6133794454928614e-69 … 1.6133794454928614e-69 3.257488532207521e-70]
 [1.0 1.0 … 1.0 1.0; 2.171116281329168e-67 4.242211772592725e-67 … 4.2422117725929514e-67 2.171116281326936e-67]
 [1.0 1.0 … 1.0 1.0; 1.6625763201282726e-65 3.0901686091382697e-65 … 3.0901686091384194e-65 1.6625763201266557e-65]
 [1.0 1.0 … 1.0 1.0; 8.166975542031823e-64 1.4586260269002753e-63 … 1.4586260269003676e-63 8.166975542024397e-64]
 [1.0 1.0 … 1.0 1.0; 2.8399535916899083e-62 4.8924063855568814e-62 … 4.892406385557177e-62 2.8399535916874495e-62]
 ⋮
 [0.8873865326184457 0.883379523960615 … 0.8833795240028807 0.8873865324932312; 0.004379028339181097 0.0051057649820871924 … 0.00510576498239587 0.004379028339717495]
 [0.8862703248585052 0.8822971914467758 … 0.88229719145664 0.886270324815552; 0.004469917091550685 0.0052094596740599715 … 0.0052094596743755515 0.004469917092254906]
 [0.8851828780492108 0.8812753084054223 … 0.8812753083687627 0.8851828781241503; 0.004558300939316953 0.0053096246485805475 … 0.005309624648913345 0.004558300940205268]
 [0.8842234502613112 0.880258884720051 … 0.8802588846987662 0.884223450297385; 0.0046402916705636146 0.005403511649273957 … 0.005403511649645809 0.004640291671402165]
 [0.8833626595066646 0.8792641983511224 … 0.8792641983846079 0.8833626594040648; 0.004717040258662972 0.005491934204305832 … 0.005491934204735164 0.0047170402592919426]
````

Here is an animation of the solution, looking only at the $v$ variable.

````julia
using CairoMakie
fig = Figure(fontsize=33)
ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"y")
tightlimits!(ax)
i = Observable(1)
u = map(i -> reshape(sol.u[i][2, :], 200, 200), i)
x = LinRange(-1, 1, 200)
y = LinRange(-1, 1, 200)
heatmap!(ax, x, y, u, colorrange=(0.0, 0.4))
hidedecorations!(ax)
record(fig, joinpath(@__DIR__, "../figures", "gray_scott_patterns.mp4"), eachindex(sol);
    framerate=60) do _i
    i[] = _i
end
````

````
"C:\\Users\\User\\.julia\\dev\\FiniteVolumeMethod\\docs\\src\\tutorials\\../figures\\gray_scott_patterns.mp4"
````

![Animation of the Gray-Scott model](../figures/gray_scott_patterns.mp4)

## Just the code
An uncommented version of this example is given below.
You can view the source code for this file [here](https://github.com/SciML/FiniteVolumeMethod.jl/tree/main/docs/src/literate_tutorials/gray_scott_model_turing_patterns_from_a_coupled_reaction_diffusion_system.jl).

```julia
using FiniteVolumeMethod, DelaunayTriangulation
tri = triangulate_rectangle(-1, 1, -1, 1, 200, 200, single_boundary=true)
mesh = FVMGeometry(tri)

bc = (x, y, t, (u, v), p) -> zero(u) * zero(v)
u_BCs = BoundaryConditions(mesh, bc, Neumann)
v_BCs = BoundaryConditions(mesh, bc, Neumann)

ε₁ = 0.00002
ε₂ = 0.00001
b = 0.04
d = 0.1
u_q = (x, y, t, α, β, γ, _ε₁) -> (-α[1] * _ε₁, -β[1] * _ε₁)
v_q = (x, y, t, α, β, γ, _ε₂) -> (-α[2] * _ε₂, -β[2] * _ε₂)
u_S = (x, y, t, (u, v), _b) -> _b * (1 - u) - u * v^2
v_S = (x, y, t, (u, v), _d) -> -_d * v + u * v^2
u_qp = ε₁
v_qp = ε₂
u_Sp = b
v_Sp = d
u_icf = (x, y) -> 1 - exp(-80 * (x^2 + y^2))
v_icf = (x, y) -> exp(-80 * (x^ 2 + y^2))
u_ic = [u_icf(x, y) for (x, y) in DelaunayTriangulation.each_point(tri)]
v_ic = [v_icf(x, y) for (x, y) in DelaunayTriangulation.each_point(tri)]
u_prob = FVMProblem(mesh, u_BCs;
    flux_function=u_q, flux_parameters=u_qp,
    source_function=u_S, source_parameters=u_Sp,
    initial_condition=u_ic, final_time=6000.0)
v_prob = FVMProblem(mesh, v_BCs;
    flux_function=v_q, flux_parameters=v_qp,
    source_function=v_S, source_parameters=v_Sp,
    initial_condition=v_ic, final_time=6000.0)
prob = FVMSystem(u_prob, v_prob)

using OrdinaryDiffEq, Sundials
sol = solve(prob, CVODE_BDF(linear_solver=:GMRES), saveat=10.0, parallel=Val(false))

using CairoMakie
fig = Figure(fontsize=33)
ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"y")
tightlimits!(ax)
i = Observable(1)
u = map(i -> reshape(sol.u[i][2, :], 200, 200), i)
x = LinRange(-1, 1, 200)
y = LinRange(-1, 1, 200)
heatmap!(ax, x, y, u, colorrange=(0.0, 0.4))
hidedecorations!(ax)
record(fig, joinpath(@__DIR__, "../figures", "gray_scott_patterns.mp4"), eachindex(sol);
    framerate=60) do _i
    i[] = _i
end
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

