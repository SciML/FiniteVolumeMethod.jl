# # Mean Exit Time on a Compound Disk and Differential-Algebraic Equations
# ```@contents
# Pages = ["mean_exit_time_on_a_compound_disk_and_differential_algebraic_equations.md"]
# ```
#
# ## Definition of the problem
# In this tutorial, we consider the problem of mean exit time, based 
# on some of my previous work.[^1] Typically, 
# mean exit time problems with linear diffusion take the form 
# ```math
# \begin{equation}\label{eq:met}
# \begin{aligned}
# D\grad^2T(\vb x) &= -1 & \vb x \in \Omega, \\
# T(\vb x) &= 0 & \vb x \in \partial \Omega, 
# \end{aligned}
# \end{equation}
# ```
# for some diffusivity $D$. $T(\vb x)$ is the mean exit time at $\vb x$, 
# meaning the average time it would take a particle starting at $\vb x$ to exit the domain 
# through $\partial\Omega$. For this interpretation of $T$, we are letting $D = \mathcal P\delta^2/(4\tau)$, 
# where $\delta > 0$ is the step length of the particle, $\tau>0$ is the duration between steps, and $\mathcal P \in [0, 1]$
# is the probability that the particle actually moves at a given time step.
#
# [^1]: See [Simpson et al. (2021)](https://iopscience.iop.org/article/10.1088/1367-2630/abe60d) and [Carr et al. (2022)](https://iopscience.iop.org/article/10.1088/1751-8121/ac4a1d).
#  In this previous work, we also use the finite volume method, but the problems are instead formulated 
#  as linear problems, which makes the solution significantly simpler to implement. The approach we give here 
#  is more generally applicable for other nonlinear problems, though.
# 
# A more complicated extension of \eqref{eq:met} is to allow the particle to be moving through 
# a _heterogenous_ media, so that the diffusivity depends on $\vb x$. In particular, 
# let us consider a compound disk $\Omega = \{0 < r < R_1\} \cup \{R_1 < r < R_2\}$, 
# and let $\mathcal P$ (the probability of movement) be piecewise constant across $\Omega$ 
# (and thus also $D$):
# ```math 
# P = \begin{cases} P_1 & 0<r<R_1,\\P_2&R_1<r<R_2,\end{cases}\quad D=\begin{cases}D_1&0<r<R_1,\\D_2&R_1<r<R_2,\end{cases}
# ```
# where $D_1 = P_1\delta^2/(4\tau)$ and $D_2=P_2\delta^2/(4\tau)$. The inner region, where 
# $0 < r < R_1$, and the outer region, where $R_1<r<R_2$, are separated by an interface 
# at $r=R_1$ and we apply an _absorbing boundary condition_ at $r=R_2$, meaning particles 
# that reach $r=R_2$ exit the domain. For this problem, \eqref{eq:met} is instead given by 
# ```math 
# \begin{equation}\label{eq:met2}
# \begin{aligned}  
# \frac{D_1}{r}\dv{r}\left(r\dv{T^{(1)}}{r}\right) &= -1 & 0 <r<R_1, \\[6pt] 
# \frac{D_2}{r}\dv{r}\left(r\dv{T^{(2)}}{r}\right) &= -1 & R_1<r<R_2,\\[6pt] 
# T^{(1)}(R_1) &= T^{(2)}(R_1), \\
# D_1\dv{T^{(1)}}{r}(R_1) &= D_2\dv{T^{(2)}}{r}(R_1), \\[6pt]
# \dv{T^{(1)}}{r}(0) &= 0, \\[6pt] 
# T^{(2)}(R_2) &= 0, 
# \end{aligned}
# \end{equation}
# ```
# which we describe using polar coordinates, and we let $T^{(1)}(r)$ 
# be the mean exit time for $0 < r < R_1$, and $T^{(2)}(r)$ be the mean exit time
# for $R_1 < r < R_2$. The boundary conditions at the interface 
# are to enforce continuity of $T$ and continuity of the flux of $T$ across 
# the interface, and the condition $\mathrm dT^{(1)}/\mathrm dr = 0$ at $r=0$ 
# is to ensure that $T^{(1)}$ is finite at $r=0$. This problem actually 
# has an exact solution, 
# ```math 
# \begin{equation}\label{eq:met2exact}
# \begin{aligned} 
# T^{(1)}(r) &= \frac{R_1^2-r^2}{4D_1}+\frac{R_2^2-R_1^2}{4D_2}, \\
# T^{(2)}(r) &= \frac{R_2^2-r^2}{4D_2},
# \end{aligned}
# \end{equation}
# ```
# which will be useful later. 
#
# One other extension we can make is to allow the interface to be more 
# complicated than just a circle. We take a _perturbed_ interface, 
# $\mathcal R_1(\theta)$, so that the inner region 
# is now $0 < r < \mathcal R_1(\theta)$ and the outer region is $\mathcal R_1(\theta) < r < R_2$.
# The function $\mathcal R_1(\theta)$ is written in the form 
# $\mathcal R_1(\theta) = R_1(1+\varepsilon g(\theta))$, where $\varepsilon \ll 1$ 
# is a perturbation parameter, $R_1$ is the radius of the unperturbed interface, 
# and $g(\theta)$ is a smooth $\mathcal O(1)$ periodic function with period $2\pi$; 
# we let $g(\theta) = \sin(3\theta) + \cos(5\theta)$ and $\varepsilon=0.05$ 
# for this tutorial. With this setup, \eqref{eq:met2} now becomes 
# ```math
# \begin{equation}\label{eq:met3}
# \begin{aligned}
# D_1\grad^2 T^{(1)}(\vb x) &= -1 & 0 < r < \mathcal R_1(\theta), \\
# D_2\grad^2 T^{(2)}(\vb x) &= -1 & \mathcal R_1(\theta) < r < R_2, \\
# T^{(1)}(\mathcal R_1(\theta),\theta) &= T^{(2)}(\mathcal R_1(\theta),\theta), \\
# D_1\grad T^{(1)}(\mathcal R_1(\theta), \theta) \vdot \vu n(\theta) &= D_2\grad T^{(2)}(\mathcal R_1(\theta), \theta) \vdot \vu n(\theta), \\
# T^{(2)}(R_2, \theta) &= 0. \\
# \end{aligned}
# \end{equation}
# ```
# This problem has no exact solution (it has a perturbation solution, though, 
# derived in [Carr et al. (2022)](https://iopscience.iop.org/article/10.1088/1751-8121/ac4a1d)).
# 
# In what follows, we consider solving both \eqref{eq:met2} and \eqref{eq:met3}.
# This will be slightly more involved than previous tutorials, though, 
# since enforcing the conditions at the interface requires that we 
# formulate the problem as a _differential-algebraic problem_. This is because 
# we cannot so easily enforce that the control volume edges lie exactly 
# on $\mathcal R_1(\theta)$ so that we can enforce, for example, 
# $D_1\grad T^{(1)}(\mathcal R_1(\theta), \theta) \vdot \vu n(\theta) = D_2\grad T^{(2)}(\mathcal R_1(\theta), \theta) \vdot \vu n(\theta)$.
# Instead, we need to look at the edges over this interface separately and apply this condition. 
# Similarly, the Dirichlet-type condition $T^{(1)}(\mathcal R_1(\theta),\theta) = T^{(2)}(\mathcal R_1(\theta),\theta)$ 
# needs to be enforced in this way, since our current support for internal Dirichlet conditions assumes that each variable is given 
# in terms of a _known_ function. We also note that, strictly speaking, a system of PDEs is not needed to represent either of \eqref{eq:met2} or \eqref{eq:met3} -
# we can just let the diffusivity vary in space. We will also discuss this point. 
# 
# To make it clear what we will actually be discussing, here is an outline:
# 1. First, we will discuss \eqref{eq:met2} on an unperturbed interface. We will solve it by representing it as a differential-algebraic problem.
# 2. Next, we discuss \eqref{eq:met3} on a perturbed interface, solved by representing it as a differential-algebraic problem.
# 3. Once we have discussed the differential-algebraic formulations, we then discuss how \eqref{eq:met2} would best be solved by using a space-varying diffusivity so that a differential-algebraic equation formulation is not needed.
# 4. Similar to (3), we discuss solving (2) without worrying about a differential-algebraic equation problem.
# 5. Finally, we discuss a specific function we have available for solving mean exit time problems with diffusivities that don't depend on $u$ so that we can instead solve them using a matrix solver.
# 
# ## Unperturbed interface: Differential-algebraic equation formulation
# Let us start by solving the problem on an unperturbed interface, namely \eqref{eq:met2}. This problem 
# can be written in a form that we are familiar with, namely \eqref{eq:met3} except with 
# $g(\theta) = 0$. The finite condition at the origin, $\mathrm dT^{(1)}/\mathrm dt = 0$, 
# is not important for this. Moreover, remember that since we have a system of PDEs, 
# we need to specify the PDEs using the flux formulation rather than the reaction-diffusion 
# formulation. In particular, $\vb q_1 = -D_1\grad T^{(1)}$ and $\vb q_2 = -D_2\grad T^{(2)}$
# are the flux functions for $T^{(1)}$ and $T^{(2)}$, and the source functions for $T^{(1)}$
# and $T^{(2)}$ are both $S=1$.
# 
# For defining the problem, we will first define the `FVMSystem` as usual, 
# without caring about the interface conditions just yet. We will then 
# discuss the differential-algebraic problem formulation. For the initial 
# conditions, which recall are the initial guesses for the solution 
# since we are solving a steady state problem, we will use 
# ```math 
# T^{(1)}(r) = \frac{R_1^2 - r^2}{4D_1}, \quad T^{(2)}(r) = \frac{R_2^2-r^2}{4D_2}.
# ```
# These come from the exact solution to the mean exit time problem on 
# a standard disk, given in [Simpson et al. (2021)](https://iopscience.iop.org/article/10.1088/1367-2630/abe60d).
# You should always try to use a good initial guess for the solution, 
# using similar problems for inspiration. 
#
# Now, let's define the mesh. We define a simple mesh first and then discuss 
# what we should do to improve it even further:
using DelaunayTriangulation, CairoMakie
R₁, R₂ = 2.0, 3.0
θ = LinRange(0, 2π, 500)
x = @. R₂ * cos(θ)
y = @. R₂ * sin(θ)
x[end] = x[begin]
y[end] = y[begin]
boundary_nodes, points = convert_boundary_points_to_indices(x, y)
tri = triangulate(points; boundary_nodes)
refine!(tri, max_area=1e-3get_total_area(tri))
triplot(tri)

# For this mesh, we also want to have some edges along the interface. 
# Let us add those in here. 
xin = @views @. R₁ * cos(θ)[begin:end-1]
yin = @views @. R₁ * sin(θ)[begin:end-1]
add_point!(tri, xin[1], yin[1])
for i in 2:length(xin)
    add_point!(tri, xin[i], yin[i])
    n = num_points(tri)
    add_edge!(tri, n - 1, n)
end
n = num_points(tri)
add_edge!(tri, n, n - length(xin) + 1) # connect the endpoints
triplot(tri)

# We should refine this mesh a bit more.
refine!(tri, max_area=1e-3get_total_area(tri))
triplot(tri)

# These constrained edges are still accessible from `get_constrained_edges(tri)`:
get_constrained_edges(tri)

# (This only returns the manually added constrained edges - thankfully, 
# it doesn't include the constrained boundary edges.) So, 
# the mesh is: 
using FiniteVolumeMethod
mesh = FVMGeometry(tri)

# Let us start by defining the boundary conditions for $T^{(1)}$. In this case, 
# there are no boundary conditions - the boundary conditions are all internal.
# We still need to provide the boundary conditions though. We provide the 
# `Constrained` type for this purpose, although this still needs a 
# function to be provided (it just won't be used). We provide some random function, 
# `sin`, for this.
T¹_BCs = BoundaryConditions(mesh, sin, Constrained)

# For $T^{(2)}$, we have absorbing boundary conditions. Remember that 
# the function takes in a `Tuple` of variables for the `u` argument.
T²_BCs = BoundaryConditions(mesh, (x, y, t, u, p) -> zero(u[2]), Dirichlet)

# Now we define the problem itself. We use mutable parameters 
# so that we can later vary the diffusivities more easily. Again, 
# remember that the flux function is provided with `Tuple`s for the 
# shape function coefficient arguments `α`, `β`, and `γ`, 
# where the `i`th elements refer to $T^{(i)}$, $i=1,2$.
D₁, D₂ = 6.25e-4, 6.25e-5
T¹_q = (x, y, t, α, β, γ, D₁) -> (-D₁[] * α[1], -D₁[] * β[1])
T²_q = (x, y, t, α, β, γ, D₂) -> (-D₂[] * α[2], -D₂[] * β[2])
T¹_qp = Ref(D₁)
T²_qp = Ref(D₂)
T¹_S = (x, y, t, u, p) -> one(u[1])
T²_S = (x, y, t, u, p) -> one(u[2])
T¹_ic_f = (x, y) -> let r = sqrt(x^2 + y^2)
    return ifelse(0 ≤ r ≤ R₁, (R₁^2 - r^2) / (4D₁), 0.0)
end
T²_ic_f = (x, y) -> let r = sqrt(x^2 + y^2)
    return (R₂^2 - r^2) / (4D₂)
end
T¹_ic = [T¹_ic_f(x, y) for (x, y) in each_point(tri)]
T²_ic = [T²_ic_f(x, y) for (x, y) in each_point(tri)]
T¹_prob = FVMProblem(mesh, T¹_BCs;
    flux_function=T¹_q, flux_parameters=T¹_qp,
    source_function=T¹_S, initial_condition=T¹_ic,
    final_time=1.0)

#-
T²_prob = FVMProblem(mesh, T²_BCs;
    flux_function=T²_q, flux_parameters=T²_qp,
    source_function=T²_S, initial_condition=T²_ic,
    final_time=1.0)

#-
prob = FVMSystem(T¹_prob, T²_prob)

# We would go further and define a `SteadyFVMProblem`, but this needs to 
# instead be applied to the `FVMDAEProblem` defined later.

# Now that we have our problem, 
# we still need to add in the constraints, which requires that 
# we use a differential-algebraic problem. In what follows, 
# some of the steps that we describe are already performed by our `DAEProblem` 
# wrapper, `FVMDAEProblem`, but it is important that you are at least 
# aware somewhat of what is done under the hood, in case you have more complicated 
# differential-algebraic equations than what we discuss here. It is 
# probably best if you read e.g. [the differential-algebraic equation example from DifferentialEquations.jl here](https://docs.sciml.ai/DiffEqDocs/stable/tutorials/dae_example/),
# and the [DifferentialEquations.jl documentation on differential-algebraic equations](https://docs.sciml.ai/DiffEqDocs/stable/types/dae_types/).
# The formulation we will be using is an implicit form, 
# in which the problem is written as 
# ```math
# \begin{equation*}
# \begin{aligned}
# \dv{u}{t} &= f(u, p, t), \\
# 0 &= g(u, p, t),
# \end{aligned}
# \end{equation*}
# ```
# where $g$ is the _constraint equation_. The function for the `DAEProblem` is now given in the form 
# `(out, du, u, p, t)`, where `out` is the residual vector. For example, if we had a problem of the form 
# ```math 
# \begin{equation*}
# \begin{aligned}
# \dv{y_1}{t} &= -0.04y_1 + 10^4y_2y_3,\\
# \dv{y_2}{t} &= 0.04y_1 - 10^4y_2y_3 - 3\times 10^7y_2^2, \\
# 1 &= y_1+y_2+y_3, 
# \end{aligned}
# \end{equation*}
# ``` 
# then our function would be 
# ```julia
# function f2(out, du, u, p, t)
#     out[1] = -0.04u[1] + 1e4 * u[2] * u[3] - du[1]
#     out[2] = +0.04u[1] - 3e7 * u[2]^2 - 1e4 * u[2] * u[3] - du[2]
#     out[3] = u[1] + u[2] + u[3] - 1.0
# end
# ```
# Notice how the `out `variable` needs to subtract off the corresponding `du` terms, 
# since `out` is the residual vector. Let us think about how we can write our problem 
# in this form. We already have an equation of the form
# ```math 
# \dv{\vb u}{t} = \vb F(\vb u, \vb p, t),
# ```
# which is just from our usual `solve` function from other tutorials. What about 
# the constraint function? Well, remember that these constraints come from 
# enforcing continuity of the solution and of the flux at the interface. So, our 
# constraint function $g$ needs to include both the Dirichlet conditions, 
# and also the Neumann conditions. Remember that the Neumann conditions are applied at 
# the edges in `get_constrained_edges(tri)`. For the Dirichlet conditions, 
# these are also applied at the edges, but they are applied by-node rather than 
# by-edge, so we should extract the unique vertices from `get_constrained_edges(tri)`
# first, 
constrained_edges = get_constrained_edges(tri)
constrained_nodes = reinterpret(reshape, Int, collect(constrained_edges)) |> unique

# We could have also just done 
constrained_nodes = first.(constrained_edges)

# since each vertex will appear at the start of an edge due to the way we constructed the edges. 
# Let us now start by defining the constraint function for the Dirichlet nodes. 
# The argument `p` in these problems will be a `Tuple`, where `p.pde_parameters` 
# gives the parameters for evaluating the PDE component, and `p.dae_parameters` 
# are for the constrained components. We will put the constrained nodes 
# into `p.dae_parameters`. This `p` will also have a field `prob` for the 
# original `FVMSystem`. Remember also that `u` in these problems 
# is a `Matrix`, with the `i`th row referring to $T^{(i)}$ and the `j`th 
# column is the value at the `j`th node. 
function dirichlet_constraints!(out, u, p, t)
    dae_params = p.dae_parameters
    cons_nodes = dae_params.constrained_nodes
    for (out_idx, node) in enumerate(cons_nodes)
        T¹ = u[1, node]
        T² = u[2, node]
        out[out_idx] = T¹ - T² # since we want T¹ = T² at the interface, the residual is T¹ - T² 
    end
    return out
end

# Now we need to consider the Neumann constraints. A useful 
# function for this is `compute_flux`, which will compute the normal
# flux across a given edge. 
function neumann_constraints!(out, u, p, t)
    prob = p.prob
    dae_params = p.dae_parameters
    cons_edges = dae_params.constrained_edges
    for (out_idx, e) in enumerate(each_edge(cons_edges))
        i, j = e
        q1, q2 = compute_flux(prob, i, j, u, t)
        out[out_idx] = q1 - q2
    end
    return out
end

# We've now defined the individual constraint functions. Putting all this 
# together, the complete constraint function is:
function constraint_function!(out, u, p, t)
    cons_nodes = p.dae_parameters.constrained_nodes
    n = length(cons_nodes)
    # The Dirichlet constraints are in out[1:n],
    # and the Neumann constraints are in out[(n+1):end]
    @views dirichlet_constraints!(out[1:n], u, p, t)
    @views neumann_constraints!(out[(n+1):end], u, p, t)
    return out
end

# For the function that gets both the PDE component and the constrained component, 
# we need to be careful with the definition of `out`. In the above, `out` 
# has been a vector. For the PDEs, though, `out` needs to be a $2\times n$ matrix, 
# where $n$ is the number of nodes. So, we should interpret `out` as a vector of 
# length $2n + m$, where $m$ is the number of constraints. 
function met_function!(out, du, u, p, t)
    # Prepare
    prob = p.prob
    tri = prob.mesh.triangulation
    n = num_points(tri)
    pde_out = @views out[1:2n]
    _pde_out = reshape(pde_out, 2, n)
    dae_out = @views out[(2n+1):end]
    pde_u = @views u[1:2n]
    _pde_u = reshape(pde_u, 2, n)
    pde_du = @views du[1:2n]
    # Solve the PDE
    pde_params = p.pde_parameters
    FiniteVolumeMethod.fvm_eqs!(_pde_out, _pde_u, pde_params, t)
    pde_out .-= pde_du
    # Evaluate the constraints 
    dae_out = @views out[(2n+1):end]
    constraint_function!(dae_out, _pde_u, p, t)
    return out
end

# We are almost there. For defining the `DAEProblem`, the call should look like 
# ```julia 
# prob = DAEProblem(prob, met_function!, du₀, u₀, tspan, p; differential_vars, kwargs...)
# ```
# Here, `du₀` are the initial values for the differentials, and `u₀` are the initial conditions, 
# with extra padding for the non-differential variables.
# We have these given as matrices from before, but `met_function!` excepts vectors, so we will need to 
# use `vec` to reshape them. We provide the function `get_dae_initial_condition` to help with obtaining 
# `du₀` and `u₀`, which will do the flattening and padding for us.
num_constraints = length(constrained_nodes) + length(constrained_edges)
u0, du0 = get_dae_initial_condition(prob, num_constraints)
u0 # T¹ is in u0[1:2:num_points(tri)], T² is in u0[2:2:2:num_points(tri)]

#-
du0

# For the `differential_vars`, we provide `get_differential_vars`.
differential_vars = get_differential_vars(prob, num_constraints)

# This is done internally inside our `DAEProblem` wrapper, `FVMDAEProblem`, but we 
# mention these variables to highlight what is going on inside `DAEProblem`, 
# and you might also need to use them if your differential-algebraic 
# is more complex. `FVMDAEProblem` also handles the callbacks needed for 
# the boundary conditions. We define this `FVMDAEProblem` below:
dae_prob = FVMDAEProblem(prob, met_function!;
    dae_parameters=(constrained_nodes=constrained_nodes, constrained_edges=constrained_edges),
    num_constraints, parallel=Val(false))

# We now have our problem. Since `SteadyStateProblem`s do not 
# yet support `DAEProblem`s, we need to apply a callback 
# manually that will terminate once the derivatives are small enough.
using Sundials
sol = solve(dae_prob, IDA())

