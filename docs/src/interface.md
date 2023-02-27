# Interface 

The definition of a PDE requires definitions for (1) the triangular mesh, (2) the boundary conditions, and (3) the PDE itself. We describe here all three of these parts, with demonstrations of them given in the Examples section. Complete docstrings can be found in the sidebar.

## FVMGeometry: Defining the mesh

The struct that defines the underlying geometry is `FVMProblem`, storing information about the mesh, the boundary, the interior, and individual information about the elements. The mesh has to be triangular, and can be constructed using my other package [DelaunayTriangulation.jl](https://github.com/DanielVandH/DelaunayTriangulation.jl). The main constructor that we provided is:

```julia
FVMGeometry(tri;
    coordinate_type=Vector{number_type(tri)},
    control_volume_storage_type_vector=NTuple{3,coordinate_type},
    control_volume_storage_type_scalar=NTuple{3,number_type(tri)},
    shape_function_coefficient_storage_type=NTuple{9,number_type(tri)},
    interior_edge_storage_type=NTuple{2,Int64},
    interior_edge_pair_storage_type=NTuple{2,interior_edge_storage_type})
```

Here, `tri` is a `Triangulation` type from DelaunayTriangulation.jl representing the mesh. An important feature in `tri` is the boundary nodes, allowing for boundary conditions to be defined. By defining these boundary nodes according to the specification in DelaunayTriangulation.jl, we can easily mix boundary conditions and different boundaries. For example, suppose we have the following domain with boundary $\Gamma_1 \cup \Gamma_2 \cup \Gamma_3 \cup \Gamma_4$:

![A segmented boundary](https://github.com/DanielVandH/FiniteVolumeMethod.jl/blob/main/test/figures/boundary_condition_example.png?raw=true)

The colours are used to distinguish between different segments of the boundaries. The boundary node vector `BNV` would thus be defined as:

```julia
Γ₁ = [2, 11, 12, 13, 14, 15, 16, 3];
Γ₂ = [3, 17, 18, 19, 20, 21, 22, 4];
Γ₃ = [4, 23, 24, 25, 26, 27, 28, 1];
Γ₄ = [1, 5, 6, 7, 8, 9, 10, 2];
BNV = [Γ₁, Γ₂, Γ₃, Γ₄]
```

It is crucial that these nodes are provided in counter-clockwise order, and that their endpoints connect (i.e. the last node of the previous segment is the same as the first node of the current segment). For inner boundaries, they are given in clockwise order, as defined in DelaunayTriangulation.jl.

A good way to generate these meshes, and the boundary nodes, is to use `generate_mesh` from [DelaunayTriangulation.jl](https://github.com/DanielVandH/DelaunayTriangulation.jl) (provided you have Gmsh installed). Note also that if you already have an existing set of triangular elements, points, and a known set of boundary nodes, the corresponding constructor for `Triangulation` (also from [DelaunayTriangulation.jl](https://github.com/DanielVandH/DelaunayTriangulation.jl)) may be of interest to you. Constrained Delaunay triangulations are in the works, but for now `generate_mesh` is sufficient.

The other keyword arguments in the function are just details about how certain variables are stored. See the docstrings in the sidebar.

## BoundaryConditions: Defining the boundary conditions

The next component to define is the set of boundary conditions, represented via the struct `BoundaryConditions`. The boundary condition functions are all assumed to take the form `f(x, y, t, u, p)`, where `p` are extra parameters that you provide. We provide the following constructor:

```julia
BoundaryConditions(mesh::FVMGeometry, functions, types;
    params=Tuple(nothing for _ in (functions isa Function ? [1] : eachindex(functions))),
    u_type=Float64, float_type=Float64)
```

Here, `functions` is a tuple for the functions for the boundary condition on each segment, where `functions[i]` should correspond to the segment the `i`th boundary segment. Then, `types` is used to declare each segment as being of *Dirichlet*, *time-dependent Dirichlet*, or *Neumann* type, with `types[i]` corresponding to the `i`th boundary segment. This variable is defined according to the rules:

```julia
is_dirichlet_type(type) = type ∈ (:Dirichlet, :D, :dirichlet, "Dirichlet", "D", "dirichlet")
is_neumann_type(type) = type ∈ (:Neumann, :N, :neumann, "Neumann", "N", "neumann")
is_dudt_type(type) = type ∈ (:Dudt, :dudt, "Dudt", "dudt", "du/dt")
```

For example, `types = (:dudt, :neumann, :D, :D)` means that the first segment has a time-dependent Dirichlet boundary condition, the second a homogeneous Neumann boundary condition, and the last two segments have Dirichlet boundary conditions (with possibly different functions). To provide the parameters `p` for each function, the keyword argument `params` is provided, letting `params[i]` be the set of parameters used when calling `functions[i]`. The type of the solution `u` can be declared using `u_type`, and the numbers representing the coordinates can be declared using `float_type`. Note that the values for any functions corresponding to a Neumann boundary condition are currently ignored (equivalent to assuming the function is zero).

## FVMProblem: Defining and solving the problem

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
    steady=false)
```

The arguments `mesh` and `boundary_conditions` are the `FVMGeometry` and `BoundaryConditions` objects defined above. The `flux_function` keyword argument must take the form `flux!(q, x, y, t, α, β, γ, p)` (`iip_flux = true`) or `flux!(x, y, t, α, β, γ, p)` (`iip_flux = false`), where `p` are some parameters (provided by `flux_parameters`) and `q` is a cache vector of size 2 (defined in the solver). If `iip_flux = false`, then the flux vector should be returned as a tuple `(q1, q2)`. The arguments `α`, `β`, and `γ` in the flux function represent the linear interpolant used, $u(x, y, t) \approx \alpha x + \beta y + \gamma$, for approximating $u$ over a single element, so that $\boldsymbol{\nabla} u(x, y, t)$ is given by $(\alpha, \beta)^{\mathsf T}$ and any instance of $u$ should be replaced by $\alpha x + \beta y + \gamma$.

If `flux_function === nothing`, then a flux function is constructed using the delay and diffusion functions (`delay_function` and `diffusion_function`, respectively), each assumed to take the form `f(x, y, t, u, p)`, with the parameters `p` given by `delay_parameters` and `diffusion_parameters` for the delay and diffusion functions, respectively. If `delay_function === nothing`, it is assumed that the delay fnuction is the identity. The flux function is constructed using the diffusion function as described at the start of the README. 

If `reaction_function === nothing`, then it is assumed that the reaction function is the zero function. Otherwise, the reaction function is assumed to take the form `f(x, y, t, u, p)`, with the parameters `p` given by `reaction_parameters`. If `delay_function !== nothing`, then this reaction function is re-defined to be `delay_function(x, y, t, u, p) * reaction_function(x, y, t, u, p)`.

The initial condition can be provided using the `initial_condition` keyword argument, and should be a vector of values so that `initial_condition[i]` is the value of `u` at `t = 0` and `(x, y) = get_point(pts, i)`.

The time span that the solution is solved over, `(initial_time, final_time)`, can be defined using the keyword arguments `initial_time` and `final_time`. 

You can solve steady problems by setting `steady = true`, in which case algorithms from NonlinearSolve.jl can be used for solving the problem with `∂u/∂t = 0`.

### Solving the FVMProblem

Once the problem has been completely defined and you now have a `prob::FVMProblem`, you are ready to solve the problem. We build on the interface provided by `DifferentialEquations.jl` (see [here](https://diffeq.sciml.ai/stable/)), using a `solve` command and any solver from `OrdinaryDiffEq.jl`. For example, we could define 
 
```julia
using OrdinaryDiffEq, LinearSlove 
alg = TRBDF2(linsolve=KLUFactorization(), autodiff=true)
```

(The code is compatible with automatic differentiation.) With this algorithm, we can easily solve the problem using 

```julia 
sol = solve(prob, alg)
```

The solution will be the same type of result returned from `OrdinaryDiffEq.jl`, with `sol.u[i]` the solution at `sol.t[i]`, and `get_point(sol.u[i], j)` is the solution at `(x, y, t) = (get_point(prob, j)..., sol.t[i])`.

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

The `parallel` keyword is supported as of v3.0. Be careful that your flux vector is thread-safe if you compute it in-place. This seems to make the code run around 4x as fast in the benchmarks I've run.

The `specialization` keyword can be used to set the specialization level for the `ODEProblem`. [See here for more details](https://diffeq.sciml.ai/stable/features/low_dep/#Controlling-Function-Specialization-and-Precompilation).

The `chunk_size` argument sets the chunk size used for automatic differentiation when defining the cache vectors. 

For examples of solving steady problems, see the Laplace's equation and Mean exit time examples in the sidebar.

## Linear Interpolants

We also provide an interface for evaluating the solutions at any point $(x, y)$, or at least evaluating the solution's associated linear interpolant. As described in the Mathematical Details section, the solution $u(x, y, t)$ is assumed to be linear inside a given triangular element $T$, i.e. $u(x, y, t) \approx \alpha(t) x + \beta(t) y + \gamma(t)$ for $(x, y) \in T$. We provide two methods for evaluating this interpolant for a given $(x, y)$ and a given $T$:
```julia
eval_interpolant(sol, x, y, t_idx::Integer, T) 
eval_interpolant(sol, x, y, t::Number, T) 
eval_interpolant!(αβγ, prob::FVMProblem, x, y, T, u)
```
The first method takes a given solution `sol` (as defined in the last section), a given coordinate `(x, y)`, an index `t_idx` such that `sol.t[t_idx]` is the time of interest, and `T` is the triangle that `(x, y)` is inside of. The second method takes in a number for the time, instead computing the solution using `sol(t)`. The third method is the one that the first and second call into, where `αβγ` is a cache vector to store the coefficients of the interpolant, `prob` is the `FVMProblem`, and `u` is the vector of the solution values (i.e. `sol.u[t_idx]`, or `sol(t)` for example). It is up to you to provide the triangle `T` that `(x, y)` is inside of, but the tools in DelaunayTriangulation.jl can make this efficient. Note that `αβγ` is stored inside `sol`, so the first and second methods do not have to create an extra cache vector on each call. 

An example of how to efficiently evaluate these interpolants is given in the examples.