# FiniteVolumeMethod

- [FiniteVolumeMethod](#finitevolumemethod)
- [Interface](#interface)
  - [The mesh](#the-mesh)
  - [The boundary conditions](#the-boundary-conditions)
  - [The problem](#the-problem)
  - [Solving the problem](#solving-the-problem)
- [Examples](#examples)
  - [Diffusion equation on a square plate](#diffusion-equation-on-a-square-plate)
  - [Diffusion equation in a wedge with mixed boundary conditions](#diffusion-equation-in-a-wedge-with-mixed-boundary-conditions)
  - [Reaction-diffusion equation with a time-dependent Dirichlet boundary condition on a disk](#reaction-diffusion-equation-with-a-time-dependent-dirichlet-boundary-condition-on-a-disk)
  - [Porous medium equation](#porous-medium-equation)

This is a package for solving partial differential equations (PDEs) of the form 

$$
\dfrac{\partial u(x, y, t)}{\partial t} + \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{q}(x, y, t, u) = R(x, y, t, u), \quad (x, y)^{\mathsf T} \in \Omega \subset \mathbb R^2,t>0,
$$

with flux and reaction functions $\boldsymbol{q}$ and $R$, respectively, using the finite volume method. The boundary conditions are assumed to take on any of the three forms:

$$
\begin{array}{rcl}
\boldsymbol{q}(x, y, t, u) \boldsymbol{\cdot} \hat{\boldsymbol{n}}(x, y) = 0, \\
\mathrm du(x, y, t)/\mathrm dt = a(x, y, t, u), \\
u(x, y, t) = a(x, y, t, u),
\end{array} \quad (x, y)^{\mathsf T} \in \partial\Omega.
$$

This first condition is a *homogeneous Neumann* boundary condition, letting $\hat{\boldsymbol{n}}(x, y)$ be the unit outward normal vector field on $\partial\Omega$ (the boundary of $\Omega$); it is possible to extend this to the inhomogeneous case, it just has not been done yet. The second condition is a *time-dependent Dirichlet* condition, and the last condition is a *Dirichlet* condition. 

An interface is also provided for solving equations of the form

$$
\frac{\partial u(x, y, t)}{\partial t} = \boldsymbol{\nabla} \boldsymbol{\cdot} \left[T(x, y, t, u)D(x, y, t, u)\boldsymbol{\nabla} u(x, y, t)\right] + T(x, y, t, u)R(x, y, t, u),
$$

where $T$ is called the *delay function*, $D$ the *diffusion function*, and $R$ the *reaction function*; the same delay is assumed to scale both diffusion and reaction. The conversion is done by noting that the corresponding flux function $\boldsymbol{q} = (q_1, q_2)^{\mathsf T}$ is simply $q_i(x, y, t, u) = -T(x, y, t, u)D(x, y, t, u)g_i$, $i=1,2$, where $(g_1, g_2)^{\mathsf T} \equiv \boldsymbol{\nabla}u(x, y, t)$ (gradients are approximated using linear interpolants; more on this in the Mathematical Details section). Similarly, the reaction function is modified so that $\tilde{R}(x, y, t, u) = T(x, y, t, u)R(x, y, t, u)$.

# Interface 

The interface for solving such PDEs requires definitions for (1) the triangular mesh, (2) the boundary conditions, and (3) the PDE itself. We describe here the details for all three parts. Note that we give examples later to illustrate all these concepts.

## The mesh

The struct that defines the underlying geometry is `FVMProblem`, storing information about the mesh, the boundary, the interior, and individual information about the elements. The mesh has to be triangular, and can be constructed using my other package [DelaunayTriangulation.jl](https://github.com/DanielVandH/DelaunayTriangulation.jl). The constructor we provide is:

```julia
FVMGeometry(T::Ts, adj, adj2v, DG, pts, BNV; 
    coordinate_type=Vector{number_type(pts)}, 
    control_volume_storage_type_vector=NTuple{3,coordinate_type}, 
    control_volume_storage_type_scalar=NTuple{3,number_type(pts)}, 
    shape_function_coefficient_storage_type=NTuple{9,number_type(pts)}, 
    interior_edge_storage_type=NTuple{2,Int64}, 
    interior_edge_pair_storage_type=NTuple{2,interior_edge_storage_type}) where {Ts}
```

Here, `T`, `adj`, `adj2v`, and `DG` are structs representing the triangles, adjacent map, adjacent-to-vertex map, and the Delaunay graph, as defined in [DelaunayTriangulation.jl](https://github.com/DanielVandH/DelaunayTriangulation.jl). The argument `pts` represents the points of the mesh, and lastly `BNV` is used to define the nodes for the separate boundary segments. For example, suppose we have the following domain with boundary $\Gamma_1 \cup \Gamma_2 \cup \Gamma_3 \cup \Gamma_4$:

![A segmented boundary](https://github.com/DanielVandH/FiniteVolumeMethod.jl/blob/main/figures/boundary_condition_example.png?raw=true)

The colours ae used to distinguish between different segments of the boundaries. The boundary node vector `BNV` would thus be defined as:

```julia
Γ₁ = [2, 11, 12, 13, 14, 15, 16, 3];
Γ₂ = [3, 17, 18, 19, 20, 21, 22, 4];
Γ₃ = [4, 23, 24, 25, 26, 27, 28, 1];
Γ₄ = [1, 5, 6, 7, 8, 9, 10, 2];
BNV = [Γ₁, Γ₂, Γ₃, Γ₄]
```

It is crucial that these nodes are provided in counter-clockwise order, and that their endpoints connect (i.e. the last node of the previous segment is the same as the first node of the current segment).

A good way to generate these meshes, and the `BNV`, is to use `generate_mesh` from [DelaunayTriangulation.jl](https://github.com/DanielVandH/DelaunayTriangulation.jl) (provided you have Gmsh installed). Note also that if you already have an existing set of triangular elements, points, and a known set of boundary nodes, the function `triangulate` (also from [DelaunayTriangulation.jl](https://github.com/DanielVandH/DelaunayTriangulation.jl)) may be of interest to you.

The other keyword arguments in the function are just details about how certain variables are stored. See `?FVMProblem` for more detail.

## The boundary conditions

The next component to define is the set of boundary conditions, represented via the struct `BoundaryConditions`. The boundary condition functions are all assumed to take the form `f(x, y, t, u, p)`, where `p` are extra parameters that you provide. We provide the following constructor:

```julia
BoundaryConditions(mesh::FVMGeometry, functions, types, boundary_node_vector;
    params=Tuple(nothing for _ in (functions isa Function ? [1] : eachindex(functions))),
    u_type=Float64, float_type=Float64)
```

Here, `functions` is a tuple for the functions for the boundary condition on each segment, where `functions[i]` should correspond to the segment represented by `BNV[i]`. Then, `types` is used to declare each segment as being of *Dirichlet*, *time-dependent Dirichlet*, or *Neumann* type, with `types[i]` corresponding to the segment represented by `BNV[i]`. This variable is defined according to the rules:

```julia
is_dirichlet_type(type) = type ∈ (:Dirichlet, :D, :dirichlet, "Dirichlet", "D", "dirichlet")
is_neumann_type(type) = type ∈ (:Neumann, :N, :neumann, "Neumann", "N", "neumann")
is_dudt_type(type) = type ∈ (:Dudt, :dudt, "Dudt", "dudt", "du/dt")
```

For example, `types = (:dudt, :neumann, :D, :D)` means that the first segment has a time-dependent Dirichlet boundary condition, the second a homogeneous Neumann boundary condition, and the last two segments have Dirichlet boundary conditions (with possibly different functions). The argument `boundary_node_vector` is the same as `BNV`. To provide the parameters `p` for each function, the keyword argument `params` is provided, letting `params[i]` be the set of parameters used when calling `functions[i]`. The type of the solution `u` can be declared using `u_type`, and the numbers representing the coordinates can be declared using `float_type`.

## The problem 

The final piece to define are the flux and reaction functions, or alternatively the diffusion, delay, and reaction functions. Moreover, the initial condition and time span must be defined. This information is represented using the struct `FVMProblem`, which has constructor:

```julia
FVMProblem(mesh, boundary_conditions;
    iip_flux=true,
    diffusion_function=nothing,
    diffusion_parameters=nothing,
    reaction_function=nothing,
    reaction_parameters=nothing,
    delay_function=nothing,
    delay_parameters=nothing,
    flux_function=nothing,
    flux_parameters=nothing,
    initial_condition,
    initial_time=0.0,
    final_time,
    steady=false,
    q_storage=SVector{2,Float64})
```

The arguments `mesh` and `boundary_conditions` are the `FVMGeometry` and `BoundaryConditions` objects defined above. The `flux_function` keyword argument must take the form `flux!(q, x, y, t, α, β, γ, p)` (`iip_flux = true`) or `flux!(x, y, t, α, β, γ, p)` (`iip_flux = false`), where `p` are some parameters (provided by `flux_parameters`) and `q` is a cache vector of size 2 (defined in the solver). If `iip_flux = false`, then you can set the storage (i.e. the vector that `q[1]` and `q[2]` are placed in) for the result `q` using the keyword argument `q_storage`. The arguments `α`, `β`, and `γ` in the flux function represent the linear interpolant used, $u(x, y, t) \approx \alpha x + \beta y + \gamma$, for approximating $u$ over a single element, so that $\boldsymbol{\nabla} u(x, y, t)$ is given by $(\alpha, \beta)^{\mathsf T}$ and any instance of $u$ should be replaced by $\alpha x + \beta y + \gamma$.

If `flux_function === nothing`, then a flux function is constructed using the delay and diffusion functions (`delay_function` and `diffusion_function`, respectively), each assumed to take the form `f(x, y, t, u, p)`, with the parameters `p` given by `delay_parameters` and `diffusion_parameters` for the delay and diffusion functions, respectively. If `delay_function === nothing`, it is assumed that the delay fnuction is the identity. The flux function is constructed using the diffusion function as described at the start of the README. 

If `reaction_function === nothing`, then it is assumed that the reaction function is the zero function. Otherwise, the reaction function is assumed to take the form `f(x, y, t, u, p)`, with the parameters `p` given by `reaction_parameters`. If `delay_function !== nothing`, then this reaction function is re-defined to be `delay_function(x, y, t, u, p) * reaction_function(x, y, t, u, p)`.

The initial condition can be provided using the `initial_condition` keyword argument, and should be a vector of values so that `initial_condition[i]` is the value of `u` at `t = 0` and `(x, y) = get_point(pts, i)`.

Finally, the time span that the solution is solved over, `(initial_time, final_time)`, can be defined using the keyword arguments `initial_time` and `final_time`. 

## Solving the problem

Once the problem has been completely defined and you now have a `prob::FVMProblem`, you are ready to solve the problem. We build on the interface provided by `DifferentialEquations.jl` (see [here](https://diffeq.sciml.ai/stable/)), using a `solve` command and any solver from `OrdinaryDiffEq.jl`. For example, we could define 
 
```julia
using OrdinaryDiffEq, LinearSlove 
alg = TRBDF2(linsolve=KLUFactorization(), autodiff=true)
```

(The code is compatible with automatic differentiation.) With this algorithm, we can easily solve the problem using 

```julia 
sol = solve(prob, alg)
```

The solution will be the same type of result returned from `OrdinaryDiffEq.jl`, with `sol.u[i]` the solution at `sol.t[i]` and at the point `(x, y) = get_point(pts, i)`.

The `solve` command is defined as follows:
```julia
SciMLBase.solve(prob::FVMProblem, alg;
    cache_eltype::Type{F}=eltype(get_initial_condition(prob)),
    jac_prototype=float.(jacobian_sparsity(prob)),
    parallel=false,
    specialization::Type{S}=SciMLBase.AutoSpecialize,
    chunk_size=PreallocationTools.ForwardDiff.pickchunksize(length(get_initial_condition(prob))),
    kwargs...) where {S,F}
```
This `cache_eltype` keyword sets the element type for the caches used for the flux vector and for `(α, β, γ)`, which is then used for wrapping a cache vector with `PreallocationTools.dualcache` for allowing automatic differentiation. 

The `jac_prototype` keyword allows for a prototype of the Jacobian to be provided. This is easy to construct with our function `jacobian_sparsity`, since the Jacobian's non-zero structure is the same as $\boldsymbol{A} + \boldsymbol{I}$, where $\boldsymbol{A}$ is the adjacency matrix of the triangulation. 

The `parallel` keyword is not currently used. One day!

The `specialization` keyword can be used to set the specialization level for the `ODEProblem`. [See here for more details](https://diffeq.sciml.ai/stable/features/low_dep/#Controlling-Function-Specialization-and-Precompilation).

The `chunk_size` argument sets the chunk size used for automatic differentiation when defining the cache vectors. 

# Examples 

Let us now give some examples. We give a variety of examples to illustrate the different ways for constructing the mesh, boundary conditions, etc. It is assumed that Gmsh is installed if you wish to run some of this code. I have set
```julia
GMSH_PATH = "./gmsh-4.9.4-Windows64/gmsh.exe"
```
We also have the following packages loaded:
```julia
using FiniteVolumeMethod
using OrdinaryDiffEq 
using LinearSolve 
using CairoMakie 
using Bessels
```

## Diffusion equation on a square plate 

We first consider the problem of diffusion on a square plate,

$$
\begin{equation*}
\begin{array}{rcll}
\displaystyle
\frac{\partial u(x, y, t)}{\partial t} &=& \dfrac19\boldsymbol{\nabla}^2 u(x, y, t) & (x, y) \in \Omega,t>0, \\
u(x, y, t) &= & 0 & (x, y) \in \partial \Omega,t>0, \\
u(x, y, 0) &= & f(x, y) &(x,y)\in\Omega,
\end{array}
\end{equation*}
$$

where $\Omega = [0, 2]^2$ and $f(x, y) = 50$ if $y \leq 1$ and $f(x, y) = 0$ if $y>1$. (This problem has an exact solution 

$$
u(x, y, t) = \dfrac{200}{\mathrm{\pi}^2}\sum_{m=1}^\infty\sum_{n=1}^\infty \frac{\left[1+(-1)^{m+1}\right]\left[1-\cos\left(\frac{n\mathrm{\pi}}{2}\right)\right]}{mn}\sin\left(\frac{m\mathrm{\pi}x}{2}\right)\sin\left(\frac{n\mathrm{\pi}y}{2}\right)\mathrm{e}^{-\frac{1}{36}\mathrm{\pi}^2(m^2+n^2)t},
$$

and we compare our results to this exact solution in the tests. See e.g. [here](http://ramanujan.math.trinity.edu/rdaileda/teach/s12/m3357/lectures/lecture_3_6_short.pdf) for a derivation of the exact solution. Comparisons not shown here.)

The first step is to define the mesh:
```julia
a, b, c, d = 0.0, 2.0, 0.0, 2.0
n = 500
x₁ = LinRange(a, b, n)
x₂ = LinRange(b, b, n)
x₃ = LinRange(b, a, n)
x₄ = LinRange(a, a, n)
y₁ = LinRange(c, c, n)
y₂ = LinRange(c, d, n)
y₃ = LinRange(d, d, n)
y₄ = LinRange(d, c, n)
x = reduce(vcat, [x₁, x₂, x₃, x₄])
y = reduce(vcat, [y₁, y₂, y₃, y₄])
xy = [[x[i], y[i]] for i in eachindex(x)]
unique!(xy)
x = getx.(xy)
y = gety.(xy)
r = 0.03
T, adj, adj2v, DG, points, BN = generate_mesh(x, y, r; gmsh_path=GMSH_PATH)
mesh = FVMGeometry(T, adj, adj2v, DG, points, BN)
```
Here I start by defining the square boundary as four segments, but then to have a single boundary segment I combine the segments into a single vector. I then create the mesh using `generate_mesh`, and then put the geometry together using `FVMGeometry`. 

Now having defined the mesh, let us define the boundary conditions. We have a homogeneous Dirichlet condition, so let us simply set
```julia
bc = ((x, y, t, u::T, p) where {T}) -> zero(T)
type = :Dirichlet # or :D or :dirichlet or "D" or "Dirichlet"
BCs = BoundaryConditions(mesh, bc, type, BN)
```

Next we must define the actual PDE. The initial condition, diffusion, and reaction functions are defined as follows:
```julia
f = (x, y) -> y ≤ 1.0 ? 50.0 : 0.0 # initial condition 
D = (x, y, t, u, p) -> 1 / 9 # You could also define flux = (q, x, y, t, α, β, γ, p) -> (q[1] = -α/9; q[2] = -β/9)
R = ((x, y, t, u::T, p) where {T}) -> zero(T)
```
Using `f`, we compute the initial condition vector:
```julia
u₀ = @views f.(points[1, :], points[2, :])
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

Now having the problem, we can solve it:
```julia
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

You can use `sol` as you would any other solution from `DifferentialEquations.jl` (e.g. `sol(t)` returns the solution at time `t`). To visualise the solution at the times `t = 0.0`, `t = 0.25`, and `t = 0.5`, the following code can be used:
```julia
using CairoMakie 
pt_mat = Matrix(points')
T_mat = [collect(T)[i][j] for i in 1:length(T), j in 1:3]
fig = Figure(resolution=(2068.72f0, 686.64f0), fontsize=38)
ax = Axis(fig[1, 1], width=600, height=600)
xlims!(ax, a, b)
ylims!(ax, c, d)
mesh!(ax, pt_mat, T_mat, color=sol.u[1], colorrange=(0, 50), colormap=:matter)
ax = Axis(fig[1, 2], width=600, height=600)
xlims!(ax, a, b)
ylims!(ax, c, d)
mesh!(ax, pt_mat, T_mat, color=sol.u[6], colorrange=(0, 50), colormap=:matter)
ax = Axis(fig[1, 3], width=600, height=600)
xlims!(ax, a, b)
ylims!(ax, c, d)
mesh!(ax, pt_mat, T_mat, color=sol.u[11], colorrange=(0, 50), colormap=:matter)
```
![Heat equation solution](https://github.com/DanielVandH/FiniteVolumeMethod.jl/blob/main/figures/heat_equation_test.png?raw=true)

## Diffusion equation in a wedge with mixed boundary conditions 

Now we consider the following problem defined on a wedge with angle $\alpha$ and mixed boundary conditions:

$$
\begin{equation*}
\begin{array}{rcll}
\dfrac{\partial u(r, \theta, t)}{\partial t} & = & \boldsymbol{\nabla}^2 u(r, \theta, t), & 0 < r < 1, 0 < \theta < \alpha, t>0, \\
\dfrac{\partial u(r, 0, t)}{\partial \theta} & = & 0, & 0 < r < 1, t>0, \\
\dfrac{\partial u(r, \alpha,t)}{\partial \theta} & = & 0, & 0 < \theta < \alpha, t>0, \\
u(r,\theta,0) & = & f(r, \theta), & 0 < r < 1, 0< \theta < \alpha.
\end{array}
\end{equation*}
$$

(The exact solution to this problem, found by writing $u(r, \theta, t) = \mathrm{e}^{-\lambda t}v(r, \theta)$ and then using separation of variables, can be shown to take the form

$$
u(r, \theta, t) = \frac12\sum_{m=1}^\infty A_{0,m}\mathrm{e}^{-\zeta_{0,m}^2t}J_0\left(\zeta_{0,m}r\right) + \sum_{n=1}^\infty\sum_{m=1}^\infty A_{n,m}\mathrm{e}^{-\zeta_{n,m}^2t}J_{n\mathrm{\pi}/\alpha}\left(\zeta_{n\mathrm{\pi}/\alpha, m}r\right)\cos\left(\frac{n\mathrm{\pi}\theta}{\alpha}\right),
$$

where, assuming $f$ can be expanded into a Fourier-Bessel series,

$$
A_{n, m} = \frac{4}{\alpha J_{n\mathrm{\pi}/\alpha + 1}^2\left(\zeta_{n\mathrm{\pi}/\alpha,m}\right)}\int_0^1\int_0^\alpha f(r, \theta)J_{n\mathrm{\pi}/\alpha}\left(\zeta_{n\mathrm{\pi}/\alpha,m}r\right)\cos\left(\frac{n\mathrm{\pi}\theta}{\alpha}\right)r\mathrm{d}r\mathrm{d}\theta, \quad n=0,1,2,\ldots,m=1,2,\ldots,
$$

and we write the roots of $J_\mu$, the $\zeta_{\mu, m}$ such that $J_\mu(\zeta_{\mu, m}) = 0$, in the form $0 < \zeta_{\mu, 1} < \zeta_{\mu, 2} < \cdots$ with $\zeta_{\mu, m} \to \infty$ as $m \to \infty$. This is the exact solution we compare to in the tests; comparisons not shown here.) We take $\alpha = \mathrm{\pi}/4$ and $f(r, \theta) = 1 - r$. 

Note that the PDE is provided in polar form, but Cartesian coordinates are assumed for the operators in our code. The conversion is easy, noting that the two Neumann conditions are just equations of the form $\boldsymbol{\nabla} u \boldsymbol{\cdot} \hat{\boldsymbol{n}} = 0$. Moreover, although the right-hand side of the PDE is given as a Laplacian, recall that $\boldsymbol{\nabla}^2 = \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{\nabla}$, so we can write $\partial u/\partial t + \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{q} = 0$, where $\boldsymbol{q} = -\boldsymbol{\nabla} u$, or `q(x, y, t, α, β, γ, p) = (-α, -β)` in the notation in our code.

Let us now solve the problem. Again, we start by defining the mesh. Since the boundary condition is different on each segment, we keep each segment as a different vector.
```julia
n = 500
α = π / 4

# The bottom edge 
r₁ = LinRange(0, 1, n)
θ₁ = LinRange(0, 0, n)
x₁ = @. r₁ * cos(θ₁)
y₁ = @. r₁ * sin(θ₁)

# Arc 
r₂ = LinRange(1, 1, n)
θ₂ = LinRange(0, α, n)
x₂ = @. r₂ * cos(θ₂)
y₂ = @. r₂ * sin(θ₂)

# Upper edge 
r₃ = LinRange(1, 0, n)
θ₃ = LinRange(α, α, n)
x₃ = @. r₃ * cos(θ₃)
y₃ = @. r₃ * sin(θ₃)

# Combine and create the mesh 
x = [x₁, x₂, x₃]
y = [y₁, y₂, y₃]
r = 0.01
T, adj, adj2v, DG, points, BN = generate_mesh(x, y, r; gmsh_path=GMSH_PATH)
mesh = FVMGeometry(T, adj, adj2v, DG, points, BN)
```

Now we define the boundary conditions.
```julia
lower_bc = ((x, y, t, u::T, p) where {T}) -> zero(T)
arc_bc = ((x, y, t, u::T, p) where {T}) -> zero(T)
upper_bc = ((x, y, t, u::T, p) where {T}) -> zero(T)
types = (:N, :D, :N)
boundary_functions = (lower_bc, arc_bc, upper_bc)
BCs = BoundaryConditions(mesh, boundary_functions, types, BN)
```

Next, the PDE itself:
```julia
f = (x, y) -> 1 - sqrt(x^2 + y^2)
D = ((x, y, t, u::T, p) where {T}) -> one(T)
u₀ = f.(points[1, :], points[2, :])
final_time = 0.1 # Do not need iip_flux = true or R(x, y, t, u, p) = 0, these are defaults 
prob = FVMProblem(mesh, BCs; diffusion_function=D, initial_condition=u₀, final_time)
```
This formulation uses the diffusion function rather than the flux function, but you could also use the flux function formulation:
```julia 
flux = (q, x, y, t, α, β, γ, p) -> (q[1] = -α; q[2] = -β; nothing)
prob2 = FVMProblem(mesh, BCs; diffusion_function=D, initial_condition=u₀, final_time)
```

Finally, we can solve and visualise the problem. The visualisation code is the essentially the same as it was for the first example, so we do not repeat it.
```julia 
alg = Rosenbrock23(linsolve=UMFPACKFactorization())
sol = solve(prob, alg; saveat=0.025)
```
![Heat equation on a wedge solution](https://github.com/DanielVandH/FiniteVolumeMethod.jl/blob/main/figures/diffusion_equation_wedge_test.png?raw=true)

## Reaction-diffusion equation with a time-dependent Dirichlet boundary condition on a disk 

Now we consider

$$
\begin{equation*}
\begin{array}{rcll}
\dfrac{\partial u(r, \theta, t)}{\partial t} & = & \boldsymbol{\nabla} \boldsymbol{\cdot} [u\boldsymbol{\nabla} u] + u(1-u), & 0 < r < 1, 0 < \theta < 2\mathrm{\pi}, \\
\dfrac{\mathrm{d}u(1, \theta, t)}{\mathrm{d}t} & = & u(1, \theta, t), & 0 < \theta < 2\mathrm{\pi}, t > 0,  \\
u(r, \theta, 0) & = & \sqrt{I_0(\sqrt{2}r)},
\end{array}
\end{equation*}
$$

where $I_0$ is the modified Bessel function of the first kind of order zero. (The solution to this problem is $u(r, \theta, t) = \mathrm{e}^t\sqrt{I_0(\sqrt{2}r)}$ (see [Bokhari et al. (2008)](https://doi.org/10.1016/j.na.2007.11.012)) This is what we compare to in the tests, and again these comparisons are not shown here.) In this case, the diffusion function is $D(x, y, t, u) = u$ and the reaction function is $R(x, y, t, u) = u(1-u)$, or equivalently the flux function is 

$$
\boldsymbol{q}(x, y, t, \alpha, \beta, \gamma) = \left(-\alpha\left(\alpha x + \beta y + \gamma\right), -\beta\left(\alpha x + \beta y + \gamma\right)\right)^{\mathsf T}. 
$$

The following code solves this problem numerically.
```julia 
## Step 1: Generate the mesh 
n = 500
r = LinRange(1, 1, 1000)
θ = LinRange(0, 2π, 1000)
x = @. r * cos(θ)
y = @. r * sin(θ)
r = 0.05
T, adj, adj2v, DG, points, BN = generate_mesh(x, y, r; gmsh_path=GMSH_PATH)
mesh = FVMGeometry(T, adj, adj2v, DG, points, BN)

## Step 2: Define the boundary conditions 
bc = (x, y, t, u, p) -> u
types = :dudt
BCs = BoundaryConditions(mesh, bc, types, BN)

## Step 3: Define the actual PDE  
f = (x, y) -> sqrt(besseli(0.0, sqrt(2) * sqrt(x^2 + y^2)))
D = (x, y, t, u, p) -> u
R = (x, y, t, u, p) -> u * (1 - u)
u₀ = [f(points[:, i]...) for i in axes(points, 2)]
final_time = 0.10
prob = FVMProblem(mesh, BCs; diffusion_function=D, reaction_function=R, initial_condition=u₀, final_time)

## Step 4: Solve
alg = FBDF(linsolve=UMFPACKFactorization())
sol = solve(prob, alg; saveat=0.025)
```
![Reaction-diffusion equation on a circle solution](https://github.com/DanielVandH/FiniteVolumeMethod.jl/blob/main/figures/reaction_diffusion_equation_test.png?raw=true)

## Porous medium equation 

We now consider the Porous medium equation,

$$
\dfrac{\partial u}{\partial t} = D\boldsymbol{\nabla} \boldsymbol{\cdot} \left[u^{m-1} \boldsymbol{u}\right],
$$

with initial condition $u(x, y, 0) = M\delta(x, y)$ where $\delta(x, y)$ is the Dirac delta function and $M = \iint_{\mathbb R^2} u(x, y, t)\mathrm{d}A$. The diffusion function here is $D(x, y, t, u) = Du^{m-1}$. We approximate $\delta(x, y)$ by 

$$
\delta(x, y) \approx g(x, y) = \frac{1}{\varepsilon^2 \mathrm{\pi}}\exp\left[-\frac{1}{\varepsilon^2}\left(x^2 + y^2\right)\right],
$$

taking $\varepsilon = 0.1$. This equation has an exact solution (see e.g. Section 17.5 of the *The porous medium equation: Mathematical theory* by J. L. Vázquez (2007)) 

$$
u(x, y, t) = \begin{cases} (Dt)^{-1/m}\left[\left(\dfrac{M}{4\mathrm{\pi}}\right)^{(m-1)/m} - \dfrac{m-1}{4m}\left(x^2+y^2\right)(Dt)^{-1/m}\right]^{1/(m-1)} & x^2 + y^2 < R_{m, M}(Dt)^{1/m}, \\
0 & x^2 + y^2 \geq R_{m, M}(Dt)^{1/m},\end{cases}
$$

where $R_{m, M} = [4m/(m-1)][M/(4\mathrm{\pi})]^{(m-1)/m}$. This equation has compact support, so we replace $\mathbb R^2$ by the domain $\Omega = [-R_{m, M}^{1/2}(DT)^{1/2m}, R_{m, M}^{1/2}(DT)^{1/2m}]^2$, where $T$ is the time that we solve up to, and we take Dirichlet boundary conditions on $\partial\Omega$. We solve this problem as follows, taking $m = 2$, $M = 0.37$, $D = 2.53$, and $T = 12$. Note the use of the parameters.
```julia
## Step 0: Define all the parameters 
m = 2
M = 0.37
D = 2.53
final_time = 12.0
ε = 0.1

## Step 1: Define the mesh 
RmM = 4m / (m - 1) * (M / (4π))^((m - 1) / m)
L = sqrt(RmM) * (D * final_time)^(1 / (2m))
n = 500
x₁ = LinRange(-L, L, n)
x₂ = LinRange(L, L, n)
x₃ = LinRange(L, -L, n)
x₄ = LinRange(-L, -L, n)
y₁ = LinRange(-L, -L, n)
y₂ = LinRange(-L, L, n)
y₃ = LinRange(L, L, n)
y₄ = LinRange(L, -L, n)
x = reduce(vcat, [x₁, x₂, x₃, x₄])
y = reduce(vcat, [y₁, y₂, y₃, y₄])
xy = [(x, y) for (x, y) in zip(x, y)]
unique!(xy)
x = getx.(xy)
y = gety.(xy)
r = 0.1
T, adj, adj2v, DG, points, BN = generate_mesh(x, y, r; gmsh_path=GMSH_PATH)
mesh = FVMGeometry(T, adj, adj2v, DG, points, BN)

## Step 2: Define the boundary conditions 
bc = ((x, y, t, u::T, p) where {T}) -> zero(T)
types = :D
BCs = BoundaryConditions(mesh, bc, types, BN)

## Step 3: Define the actual PDE  
f = (x, y) -> M * 1 / (ε^2 * π) * exp(-1 / (ε^2) * (x^2 + y^2))
diff_fnc = (x, y, t, u, p) -> p[1] * u^(p[2] - 1)
diff_parameters = (D, m)
u₀ = [f(points[:, i]...) for i in axes(points, 2)]
prob = FVMProblem(mesh, BCs; diffusion_function=diff_fnc,
    diffusion_parameters=diff_parameters, initial_condition=u₀, final_time)

## Step 4: Solve
alg = TRBDF2(linsolve=KLUFactorization())
sol = solve(prob, alg; saveat=3.0)
```
![Porous-medium equation with m=2](https://github.com/DanielVandH/FiniteVolumeMethod.jl/blob/main/figures/porous_medium_test.png?raw=true)

We can continue this example with the Porous medium equation by considering the same equation except with a linear source:

$$
\dfrac{\partial u}{\partial t} = D\boldsymbol{\nabla} \boldsymbol{\cdot} \left[u^{m-1}\boldsymbol{\nabla} u\right] + \lambda u, \quad \lambda>0. 
$$

This equation has an exact solution given by 

$$
u(x, y, t) = \mathrm{e}^{\lambda t}v\left(x, y, \frac{D}{\lambda(m-1)}\left[\mathrm{e}^{\lambda(m-1)t} - 1\right]\right),
$$

where $u(x, y, 0) = M\delta(x, y)$ and $v$ is the exact solution we gave above except with $D=1$. This is what we use for assessing the solution in the tests - not shown here. The domain we use is now $\Omega = [-R_{m, M}^{1/2}\tau(T)^{1/2m}, R_{m,M}^{1/2}\tau(T)^{1/2m}]^2$, where $\tau(T) = \frac{D}{\lambda(m-1)}[\mathrm{e}^{\lambda(m-1)T}-1]$. The code below solves this problem.
```julia
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
n = 500
x₁ = LinRange(-L, L, n)
x₂ = LinRange(L, L, n)
x₃ = LinRange(L, -L, n)
x₄ = LinRange(-L, -L, n)
y₁ = LinRange(-L, -L, n)
y₂ = LinRange(-L, L, n)
y₃ = LinRange(L, L, n)
y₄ = LinRange(L, -L, n)
x = reduce(vcat, [x₁, x₂, x₃, x₄])
y = reduce(vcat, [y₁, y₂, y₃, y₄])
xy = [(x, y) for (x, y) in zip(x, y)]
unique!(xy)
x = getx.(xy)
y = gety.(xy)
r = 0.07
T, adj, adj2v, DG, points, BN = generate_mesh(x, y, r; gmsh_path=GMSH_PATH)
mesh = FVMGeometry(T, adj, adj2v, DG, points, BN)

## Step 2: Define the boundary conditions 
bc = ((x, y, t, u::T, p) where {T}) -> zero(T)
types = :D
BCs = BoundaryConditions(mesh, bc, types, BN)

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
```
![Porous-medium equation with linear source](https://github.com/DanielVandH/FiniteVolumeMethod.jl/blob/main/figures/porous_medium_linear_source_test.png?raw=true)
