# FiniteVolumeMethod

- [FiniteVolumeMethod](#finitevolumemethod)
- [Interface](#interface)
  - [The mesh](#the-mesh)
  - [The boundary conditions](#the-boundary-conditions)
  - [The problem](#the-problem)
  - [Solving the problem](#solving-the-problem)
  - [Linear interpolants](#linear-interpolants)
- [Examples](#examples)
  - [Diffusion equation on a square plate](#diffusion-equation-on-a-square-plate)
  - [Diffusion equation in a wedge with mixed boundary conditions](#diffusion-equation-in-a-wedge-with-mixed-boundary-conditions)
  - [Reaction-diffusion equation with a time-dependent Dirichlet boundary condition on a disk](#reaction-diffusion-equation-with-a-time-dependent-dirichlet-boundary-condition-on-a-disk)
  - [Porous medium equation](#porous-medium-equation)
  - [Porous-Fisher equation and travelling waves](#porous-fisher-equation-and-travelling-waves)
  - [Using the linear interpolants](#using-the-linear-interpolants)
- [Mathematical Details](#mathematical-details)
  - [Interior discretisation](#interior-discretisation)
  - [Boundary conditions](#boundary-conditions)

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

![A segmented boundary](https://github.com/DanielVandH/FiniteVolumeMethod.jl/blob/main/test/figures/boundary_condition_example.png?raw=true)

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

For example, `types = (:dudt, :neumann, :D, :D)` means that the first segment has a time-dependent Dirichlet boundary condition, the second a homogeneous Neumann boundary condition, and the last two segments have Dirichlet boundary conditions (with possibly different functions). The argument `boundary_node_vector` is the same as `BNV`. To provide the parameters `p` for each function, the keyword argument `params` is provided, letting `params[i]` be the set of parameters used when calling `functions[i]`. The type of the solution `u` can be declared using `u_type`, and the numbers representing the coordinates can be declared using `float_type`. Note that the values for any functions corresponding to a Neumann boundary condition are currently ignored (equivalent to assuming the function is zero).

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

## Linear interpolants 

We also provide an interface for evaluating the solutions at any point $(x, y)$, or at least evaluating the solution's associated linear interpolant. As described in the Mathematical Details section, the solution $u(x, y, t)$ is assumed to be linear inside a given triangular element $T$, i.e. $u(x, y, t) \approx \alpha(t) x + \beta(t) y + \gamma(t)$ for $(x, y) \in T$. We provide two methods for evaluating this interpolant for a given $(x, y)$ and a given $T$:
```julia
eval_interpolant(sol, x, y, t_idx::Integer, T) 
eval_interpolant(sol, x, y, t::Number, T) 
eval_interpolant!(αβγ, prob::FVMProblem, x, y, T, u)
```
The first method takes a given solution `sol` (as defined in the last section), a given coordinate `(x, y)`, an index `t_idx` such that `sol.t[t_idx]` is the time of interest, and `T` is the triangle that `(x, y)` is inside of. The second method takes in a number for the time, instead computing the solution using `sol(t)`. The third method is the one that the first and second call into, where `αβγ` is a cache vector to store the coefficients of the interpolant, `prob` is the `FVMProblem`, and `u` is the vector of the solution values (i.e. `sol.u[t_idx]`, or `sol(t)` for example). It is up to you to provide the triangle `T` that `(x, y)` is inside of, but the tools in DelaunayTriangulation.jl can make this efficient. Note that `αβγ` is stored inside `sol`, so the first and second methods do not have to create an extra cache vector on each call. 

An example of how to efficiently evaluate these interpolants is given in the Examples section.

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
using Test
```
See the tests for more detail.

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
![Heat equation solution](https://github.com/DanielVandH/FiniteVolumeMethod.jl/blob/main/test/figures/heat_equation_test.png?raw=true)

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
A_{n, m} = \frac{4}{\alpha J_{n\mathrm{\pi}/\alpha + 1}^2\left(\zeta_{n\mathrm{\pi}/\alpha,m}\right)}\int_0^1\int_0^\alpha f(r, \theta)J_{n\mathrm{\pi}/\alpha}\left(\zeta_{n\mathrm{\pi}/\alpha,m}r\right)\cos\left(\frac{n\mathrm{\pi}\theta}{\alpha}\right)r~\mathrm{d}r~\mathrm{d}\theta, \quad n=0,1,2,\ldots,m=1,2,\ldots,
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
![Heat equation on a wedge solution](https://github.com/DanielVandH/FiniteVolumeMethod.jl/blob/main/test/figures/diffusion_equation_wedge_test.png?raw=true)

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
![Reaction-diffusion equation on a circle solution](https://github.com/DanielVandH/FiniteVolumeMethod.jl/blob/main/test/figures/reaction_diffusion_equation_test.png?raw=true)

## Porous medium equation 

We now consider the Porous medium equation,

$$
\dfrac{\partial u}{\partial t} = D\boldsymbol{\nabla} \boldsymbol{\cdot} \left[u^{m-1} \boldsymbol{u}\right],
$$

with initial condition $u(x, y, 0) = M\delta(x, y)$ where $\delta(x, y)$ is the Dirac delta function and $M = \iint_{\mathbb R^2} u(x, y, t)~\mathrm{d}A$. The diffusion function here is $D(x, y, t, u) = Du^{m-1}$. We approximate $\delta(x, y)$ by 

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
![Porous-medium equation with m=2](https://github.com/DanielVandH/FiniteVolumeMethod.jl/blob/main/test/figures/porous_medium_test.png?raw=true)

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
![Porous-medium equation with linear source](https://github.com/DanielVandH/FiniteVolumeMethod.jl/blob/main/test/figures/porous_medium_linear_source_test.png?raw=true)

## Porous-Fisher equation and travelling waves 

We now consider a more involved example, where we discuss the travelling wave solutions of the Porous-Fisher equation and discuss how we test a more complicated problem. We consider:

$$
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
$$

This problem is defined on the rectangle $[0, a] \times [0, b]$, and we assume that $b \gg a$ so that the rectangle is much taller than it is wide. This problem has $u$ ranging from $u=1$ at the bottom of the rectangle down to $u=0$ at the top of the rectangle, with no flux conditions on the two vertical walls. The function $f(y)$ is taken to be independent of $x$. This setup implies that the solution along each constant line $x = x_0$ should be about the same, i.e. the problem is invariant in $x$. If indeed we have $u(x, y, t) \equiv u(y, t)$, then the PDE becomes

$$
\dfrac{\partial u(y, t)}{\partial t} = D\dfrac{\partial}{\partial y}\left(u\dfrac{\partial u}{\partial y}\right) + \lambda u(1-u),
$$

which has travelling wave solutions. Following the analysis from Section 13.4 of *Mathematical biology I: An introduction* by J. D. Murray (2002), we can show that a travelling wave solution to the one-dimensional problem above is

$$
u(y, t) = \left(1 - \mathrm{e}^{c_{\min}z}\right)\left[z \leq 0\right]
$$

where $c_{\min} = \sqrt{\lambda/(2D)}$, $c = \sqrt{D\lambda/2}$, and $z = x - ct$ is the travelling wave coordinate. This travelling wave would match our problem exactly if the rectangle were instead $[0, a] \times \mathbb R$, but by choosing $b$ large enough we can at least emulate the travelling wave behaviour closely; homogeneous Neumann conditions are to ensure no energy is lost, thus allowing the travelling waves to exist. Note also that the approximations of the solution with $u(y, t)$ above will only be accurate for large time.

Let us now solve the problem. For this problem, rather than using `generate_mesh` we will use a structured triangulation with `triangulate_structured`. This will make it easier to test the $x$ invariance.

```julia
## Step 1: Define the mesh 
a, b, c, d, Nx, Ny = 0.0, 3.0, 0.0, 40.0, 60, 80
T, adj, adj2v, DG, points, BN = triangulate_structured(a, b, c, d, Nx, Ny; return_boundary_types=true)
mesh = FVMGeometry(T, adj, adj2v, DG, points, BN)

## Step 2: Define the boundary conditions 
a₁ = ((x, y, t, u::T, p) where {T}) -> one(T)
a₂ = ((x, y, t, u::T, p) where {T}) -> zero(T)
a₃ = ((x, y, t, u::T, p) where {T}) -> zero(T)
a₄ = ((x, y, t, u::T, p) where {T}) -> zero(T)
bc_fncs = (a₁, a₂, a₃, a₄)
types = (:D, :N, :D, :N)
BCs = BoundaryConditions(mesh, bc_fncs, types, BN)

## Step 3: Define the actual PDE  
f = ((x::T, y::T) where {T}) -> zero(T)
diff_fnc = (x, y, t, u, p) -> p * u
reac_fnc = (x, y, t, u, p) -> p * u * (1 - u)
D, λ = 0.9, 0.99
diff_parameters = D
reac_parameters = λ
final_time = 50.0
u₀ = [f(points[:, i]...) for i in axes(points, 2)]
prob = FVMProblem(mesh, BCs; diffusion_function=diff_fnc, reaction_function=reac_fnc,
    diffusion_parameters=diff_parameters, reaction_parameters=reac_parameters,
    initial_condition=u₀, final_time)

## Step 4: Solve
alg = TRBDF2(linsolve=KLUFactorization())
sol = solve(prob, alg; saveat=0.5)
```

This gives us our solution. To verify the $x$-invariance, something like the following suffices:
```julia
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
The results we obtain are shown below,with the exact travelling wave from the one-dimensional problem shown in red in the fourth plot and the numerical solutions are the other curves. 

![Travelling wave problem](https://github.com/DanielVandH/FiniteVolumeMethod.jl/blob/main/test/figures/travelling_wave_problem_test.png?raw=true)

## Using the linear interpolants 

We now give an example of how one can efficiently evaluate the linear interpolants for a given solution. We illustrate this using the porous medium equation with a linear source example. Letting `prob` be as we computed in that example, we find the solution:

```julia
alg = TRBDF2(linsolve=KLUFactorization(), autodiff=true)
sol = solve(prob, alg, saveat=2.5)
```

If we just have a single point $(x, y)$ to evaluate the interpolant at, for a given $t$, we can do the following. First, we define the triple $(x, y, t)$:

```julia
x = 0.37 
y = 0.58
t_idx = 5 # t = sol.t[t_idx]
```

Next, we must find what triangle contains `(x, y)`. This is done by calling into the point location method provided by DelaunayTriangulation.jl, namely `jump_and_march`. We provide a simple interface for this using `FVMProblem`, which we use as follows:
```julia
V = jump_and_march(x, y, prob)
@test DelaunayTriangulation.isintriangle(get_point(points, V...)..., (x, y)) == 1
```
(You can also provide keyword arguments to `jump_and_march`, matching those from DelaunayTriangulation.jl.) Now we can evaluate the interpolant at this point:
```
val = eval_interpolant(sol, x, y, t_idx, V)
# or eval_interpolant(sol, x, y, sol.t[t_idx], V)
```
This is our approximation to $u(0.37, 0.58, 0.2)$.

A more typical example would involve evaluating this interpolant over a much larger set of points. A good way to do this is to first find all the triangles that correspond to each point. In what follows, we define a lattice of points, and then we find the triangle for each point. To accelerate the procedure, when initiating the `jump_and_march` function we will tell it to also try starting at the previously found triangle. Note that we also put the grid slightly off the boundary since the generated mesh doesn't exactly lie on the square $[0, 2]^2$, hence some points wouldn't be in any triangle if we put some points exactly on this boundary.
```julia
nx = 250
ny = 250
grid_x = LinRange(-L + 1e-1, L - 1e-1, nx)
grid_y = LinRange(-L + 1e-1, L - 1e-1, ny)
V_mat = Matrix{NTuple{3, Int64}}(undef, nx, ny)
last_triangle = rand(FVM.get_elements(prob)) # initiate 
for j in 1:ny 
    for i in 1:nx 
        V_mat[i, j] = jump_and_march(grid_x[i], grid_y[j], prob; try_points = last_triangle)
        last_triangle = V_mat[i, j]
    end
end
```

Now let's evaluate the interpolant at each time.
```julia 
u_vals = zeros(nx, ny, length(sol))
for k in eachindex(sol)
    for j in 1:ny
        for i in 1:nx
            V = V_mat[i, j]
            u_vals[i, j, k] = eval_interpolant(sol, grid_x[i], grid_y[j], k, V)
        end
    end
end
```

This setup now makes it easy to use `surface!` from Makie.jl to visualise the solution, thanks to our regular grid.
```julia
fig = Figure(resolution=(2744.0f0, 692.0f0))
for k in 1:4
    ax = Axis3(fig[1, k])
    zlims!(ax, 0, 1), xlims!(ax, -L - 1e-1, L + 1e-1), ylims!(ax, -L - 1e-1, L + 1e-1)
    surface!(ax, grid_x, grid_y, u_vals[:, :, k+1], colormap=:matter)
end 
```
![Surface plots](https://github.com/DanielVandH/FiniteVolumeMethod.jl/blob/main/test/figures/surface_plots_travelling_wave.png?raw=true)

# Mathematical Details

We now describe the mathematical details involved with the finite volume method as we have applied it. We assume that we have some triangulation of $\Omega$, like a (constrained) Delaunay triangulation $\mathcal D\mathcal T(\Omega)$. 

## Interior discretisation

This triangulation is used to define control volumes around each point. This is illustrated in the following figure, where (a) shows the domain $\Omega$ and its triangulation $\mathcal T(\Omega)$, together with the boundary $\partial\Omega$ shown in blue. (b) shows the mesh in (a) along with the dual mesh shown in blue, with red points showing the centroids of each triangle in $\mathcal{T}(\Omega)$. The blue polygons around each nodal point are the control volumes, and we denote the control volume around some point $\boldsymbol{x}_i$ by $\Omega_i$ and its boundary is $\partial\Omega_i$. (Note that this is the so-called ``vertex-centred approach'' to the finite volume method.)

![Dual mesh](https://github.com/DanielVandH/FiniteVolumeMethod.jl/blob/main/test/figures/triangulation_example.png?raw=true)

To be more precise, consider some interior point $\boldsymbol{x}\_{i}  = (x\_i, y\_i)^{\mathsf T} \in \Omega$ which is a point in $\mathcal T(\Omega)$, i.e. one of the black points on the figure above. We take the centroids of the neighbouring triangles of $\boldsymbol{x}\_i$ and connect these centroids to the midpoints of the associated triangle. These connections defined a closed polygon around $\boldsymbol{x}\_i$ which we denote by $\partial\Omega_i$, and its interior will be $\Omega\_i$ with some volume $V\_i$. This polygon will be comprised of a set of edges $\mathcal E_i$, and for each $\boldsymbol{x}\_{\sigma} \in\mathcal E\_i$ there will be a length $L\_\sigma$, a midpoint $\boldsymbol{x}\_{\sigma}$, and a unit normal vector $\hat{\boldsymbol{n}}\_{i, \sigma}$ which is normal to $\sigma$ and directed outwards to $\Omega\_i$ with $\|\hat{\boldsymbol{n}}\_{i,\sigma}\| = 1$. This notation is elucidated in the figure below. In this figure, the green area shows $\Omega_i$, and its boundary $\partial\Omega_i$ is in blue. The edge $\sigma$ is shown in blue. Lastly, the cyan points show an example ordering $(v_{k1}, v_{k2}, v_{k3})$ of a triangle $T_k \in \mathcal T(\Omega)$. It is with these control volumes that we can now discretise our PDE $\partial_t u + \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{q} = R$.

![Control volume](https://github.com/DanielVandH/FiniteVolumeMethod.jl/blob/main/test/figures/control_volume_example.png?raw=true)

Let us start by integrating our equations around $\Omega\_i$ and moving the time-derivative outside the integral:

$$
\begin{equation}
\dfrac{\mathrm d}{\mathrm dt}\iint_{\Omega_i} u~ \mathrm{d}V + \iint_{\Omega_i} \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{q}~ \mathrm{d}V = \iint_{\Omega_i} R~ \mathrm{d}V. 
\end{equation} 
$$

Using the divergence theorem, the second integral becomes

$$
\begin{equation}
\iint_{\Omega_i}  \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{q}~\mathrm{d}V = \oint_{\partial\Omega_i} \boldsymbol{q} \boldsymbol{\cdot} \hat{\boldsymbol{n}}_{i, \sigma}~\mathrm{d}s = \sum_{\sigma\in \mathcal E_i} \int_\sigma \boldsymbol{q} \boldsymbol{\cdot} \hat{\boldsymbol{n}}_{i, \sigma}~\mathrm{d}s,
\end{equation}
$$

where the last equality follows from integrating over each individual line segment that defines $\partial\Omega_i$. We now define the control volume averages,

$$
\begin{equation}
\bar u_i = \frac{1}{V_i}\iint_{\Omega_i} u~\mathrm dV, \qquad \bar R_i = \frac{1}{V_i}\iint_{\Omega_i} R~\mathrm dV,
\end{equation}
$$

so that our integral formulation becomes

$$
\begin{equation}
\dfrac{\mathrm d\bar u_i}{\mathrm dt} + \frac{1}{V_i}\sum_{\sigma\in\mathcal E_i}\int_{\sigma} \boldsymbol{q} \boldsymbol{\cdot} \hat{\boldsymbol{n}}_{i, \sigma}~\mathrm ds = \bar R_i. 
\end{equation}
$$

The line integrals can be approximated using a midpoint rule, 

$$
\begin{equation}
\int_\sigma \boldsymbol{q} \boldsymbol{\cdot} \hat{\boldsymbol{n}}\_{i, \sigma}~\mathrm ds \approx \left[\boldsymbol{q}(x\_\sigma, y\_\sigma, t, u) \boldsymbol{\cdot} \hat{\boldsymbol{n}}\_{i, \sigma}\right] L_\sigma. 
\end{equation} 
$$

Lastly, the control volume averages can be approximated by simply replacing them by their value at $(x\_i, y\_i)$. We thus obtain the following approximation, where $\tilde u\_i$ denotes an approximation to the exact solution $u$ at $(x_i, y_i)$:

$$ 
\begin{equation} 
\frac{\mathrm d\tilde u_i}{\mathrm dt} + \frac{1}{V_i}\sum_{\sigma \in \mathcal E_i} \left[\boldsymbol{q}(x_\sigma, y_\sigma, t, u) \boldsymbol{\cdot} \hat{\boldsymbol{n}}\_{i, \sigma}\right] L\_\sigma = \tilde R_i,
\end{equation} 
$$

where $\tilde R\_i$ is the approximation to $\bar R\_i$ at $(x\_i, y\_i)$. This approximation is what we use in the interior of $\Omega$ for approximating the value of $u$ at each node. 

We still need to discuss how we compute $\boldsymbol{q}(x\_{\sigma}, y\_{\sigma}, t, u)$. To deal with this function, let $\mathcal T\_i$ be the set of triangles in $\mathcal T(\Omega)$ that have $\boldsymbol{x}\_i$ as a node, and consider a triangle $T\_k \in \mathcal T\_i$. We will inteprolate $\tilde u$ with a linear shape function in $T_k$, so that 

$$ 
\begin{equation}
\tilde u(x, y, t) = \alpha_k x + \beta_ky + \gamma_k, \quad (x, y) \in T_k.
\end{equation} 
$$

(The dependence of the coefficients $\alpha\_k$, $\beta\_k$, and $\gamma\_k$ on $t$ is suppressed.) We suppose that the three nodes defining $T_k$ are $(v\_{k1}, v\_{k2}, v\_{k3})$, and these points are given in a counter-clockwise order. An example of how these points are defined was shown in the control volume schematic figure. We can find $\alpha\_k$, $\beta\_k$, and $\gamma\_k$ by noting that we know the values of $\tilde u$ at the nodes $v\_{k1}$, $v\_{k2}$, and $v\_{k3}$, either from the initial condition or from the previous time-step of the integration. We will denote the value of $\tilde u$ at $v\_{ki}$ by $\tilde u\_{v_{ki}}$, $i=1,2,3$. We then have the system of equations, 

$$ 
\begin{equation} 
\tilde u_{v_{ki}} = \alpha_k x_{v_{ki}} + \beta_k y_{v_{ki}} + \gamma_k, \quad i=1,2,3, 
\end{equation} 
$$ 

where $x_{v_{ki}}$ and $y_{v_{ki}}$ denote the $x$- and $y$-coordinates of the point $v\_{ki}$, respectively, for $i=1,2,3$. The system can be written in matrix form as 

```math
\begin{bmatrix} 
x_{v_{k1}} & y_{v_{k1}} & 1 \\ x_{v_{k2}} & y_{v_{k2}} & 1 \\ x_{v_{k3}} & y_{v_{k3}} & 1 \end{bmatrix}\begin{bmatrix} \alpha_k \\ \beta_k \\ \gamma_k \end{bmatrix} = \begin{bmatrix} \tilde u_{v_{k1}} \\ \tilde u_{v_{k2}} \\ \tilde u_{v_{k3}}
\end{bmatrix} 
```

Using Cramer's rule, we can define 

$$
\begin{equation} 
\begin{array}{lcllcllcl}
s_{k1} &=& \dfrac{y_{v_{k2}} - y_{v_{k3}}}{\Delta_k}, & s_{k2} &=& \dfrac{y_{v_{k3}}-y_{v_{k1}}}{\Delta_k}, & s_{k3} &=& \dfrac{y_{v_{k1}}-y_{v_{k2}}}{\Delta_k}, \\
s_{k4} &=& \dfrac{x_{v_{k3}}-x_{v_{k2}}}{\Delta_k},& s_{k5} &=& \dfrac{x_{v_{k1}}-x_{v_{k3}}}{\Delta_k}, & s_{k6} &=& \dfrac{x_{v_{k2}}-x_{v_{k1}}}{\Delta_k}, \\
s_{k7} &=& \dfrac{x_{v_{k2}}y_{v_{k3}}-x_{v_{k3}}y_{v_{k2}}}{\Delta_k}, & s_{k8} &=& \dfrac{x_{v_{k3}}y_{v_{k1}}-x_{v_{k1}}y_{v_{k3}}}{\Delta_k},& s_{k9} &= &\dfrac{x_{v_{k1}}y_{v_{k2}}-x_{v_{k2}}y_{v_{k1}}}{\Delta_k}, 
\end{array} 
\end{equation} 
$$

and 

$$ 
\begin{equation}
\Delta_k = x_{v_{k1}}y_{v_{k2}}-x_{v_{k2}}y_{v_{k1}}-x_{v_{k1}}y_{v_{k3}}+x_{v_{k3}}y_{v_{k1}}+x_{v_{k2}}y_{v_{k3}}-x_{v_{k3}}y_{v_{k2}}.
\end{equation}
$$

With this notation, 

$$
\begin{equation} 
\begin{array}{rcl}
\alpha_k &=& s_{k1}\tilde u_{v_{k1}} + s_{k2}\tilde u_{v_{k2}} + s_{k3}\tilde u_{v_{k3}}, \\
\beta_k &= & s_{k4}\tilde u_{v_{k1}} + s_{k5}\tilde u_{v_{k2}} + s_{k6}\tilde u_{v_{k3}}, \\
\gamma_k &=& s_{k7}\tilde u_{v_{k1}} + s_{k8}\tilde u_{v_{k2}} + s_{k9}\tilde u_{v_{k3}}.
\end{array}
\end{equation} 
$$ 

With these coefficients, our approximation becomes 

$$ 
\begin{equation} 
\frac{\mathrm du_i}{\mathrm dt} + \frac{1}{V_i}\sum_{\sigma\in\mathcal E_i} \left[\boldsymbol{q}\left(x_\sigma, y_\sigma, t, \alpha_{k(\sigma)}x_\sigma + \beta_{k(\sigma)}y_\sigma + \gamma_{k(\sigma)}\right) \boldsymbol{\cdot} \hat{\boldsymbol{n}}\_{i, \sigma}\right] L\_\sigma = R_i,
\end{equation} 
$$ 

where we now drop the tilde notation and make the approximations implicit, and now the $k(\sigma)$ notation is used to refer to the edge $\sigma$ inside triangle $T_{k(\sigma)}$. This linear shape function also allows to compute gradients like $\boldsymbol{\nabla} u(x_\sigma, y_\sigma)$, since $\boldsymbol{\nabla} u(x_\sigma, y_\sigma) = (\alpha_{k(\sigma)}, \beta_{k(\sigma)})^{\mathsf T}$.

## Boundary conditions 

As discussed at the start, we only support boundary conditions of the form 

$$
\begin{array}{rcl}
\boldsymbol{q}(x, y, t, u) \boldsymbol{\cdot} \hat{\boldsymbol{n}}(x, y) = 0, \\
\mathrm du(x, y, t)/\mathrm dt = a(x, y, t, u), \\
u(x, y, t) = a(x, y, t, u),
\end{array} \quad (x, y)^{\mathsf T} \in \partial\Omega.
$$

For the Neumann boundary condition, recall that the integral form of our PDE was 

$$
\begin{equation}
\dfrac{\mathrm d\bar u_i}{\mathrm dt} + \frac{1}{V_i}\sum_{\sigma\in\mathcal E_i}\int_{\sigma} \boldsymbol{q} \boldsymbol{\cdot} \hat{\boldsymbol{n}}_{i, \sigma}~\mathrm ds = \bar R_i. 
\end{equation}
$$

Thus, if $\sigma$ is an edge such that $\boldsymbol{q} \boldsymbol{\cdot} \hat{\boldsymbol{n}} = 0$, then the contribution from $\sigma$ to the above sum is zero. Thus, in our code, we simply skip over such a $\sigma$ when computing the sum. 

For the time-dependent Dirichlet boundary condition, we can skip over nodes with this condition and simply set $\mathrm du\_i/\mathrm dt = a(x, y, t, u)$. 

Lastly, for the Dirichlet boundary conditions, we leave $\mathrm du\_i/\mathrm dt = 0$ and simply update the value of $u\_i$ with $a(x, y, t, u\_i)$ at the end of each iteration. This is done using callbacks.
