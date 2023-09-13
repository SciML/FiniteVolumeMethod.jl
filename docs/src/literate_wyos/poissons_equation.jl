# # Poisson's Equation 
# ```@contents 
# Pages = ["poissons_equation.md"]
# ``` 
# We now write a solver for Poisson's equation. What we produce 
# in this section can also be accessed in `FiniteVolumeMethod.PoissonsEquation`.

# ## Mathematical Details
# We start by describing the mathematical details. The problems we will be solving 
# take the form 
# ```math 
# \div[D(\vb x)\grad u] = f(\vb x).
# ``` 
# Note that this is very similar to a mean exit time problem, except 
# $f(\vb x) = -1$ for mean exit time problems. Note that this is actually 
# a generalised Poisson equation - typically these equations look like 
# $\grad^2 u = f$.[^1]
# [^1]: See, for example, [this paper](https://my.ece.utah.edu/~ece6340/LECTURES/Feb1/Nagel%202012%20-%20Solving%20the%20Generalized%20Poisson%20Equation%20using%20FDM.pdf).
#
# From these similarities, we already know that 
# ```math 
# \frac{1}{V_i}\sum_{\sigma\in\mathcal E_i} D(\vb x_\sigma)\left[\left(s_{k, 11}n_\sigma^x+s_{k,21}n_\sigma^y\right)u_{k1} + \left(s_{k,12}n_\sigma^x+s_{k,22}n_\sigma^y\right)u_{k2}+\left(s_{k,13}n_\sigma^x+s_{k,23}n_\sigma^y\right)u_{k3}\right]L_\sigma = f(\vb x_i),
# ```
# and thus we can write this as $\vb a_i^{\mkern-1.5mu\mathsf T}\vb u = b_i$ as usual, with $b_i = f(\vb x_i)$.
# The boundary conditions are handled in the same way as in [mean exit time problems](mean_exit_time.md).

# ## Implementation
# Let us now implement our problem. For [mean exit time problems](mean_exit_time.md), we
# had a function `create_met_b!` that we used for defining $\vb b$. We should generalised 
# that function to now accept a source function:
function create_rhs_b!(A, mesh, conditions, source_function, source_parameters)
    b = zeros(DelaunayTriangulation.num_solid_vertices(mesh.triangulation))
    for i in each_solid_vertex(mesh.triangulation)
        if !FVM.is_dirichlet_node(conditions, i)
            p = get_point(mesh, i)
            x, y = getxy(p)
            b[i] = source_function(x, y, source_parameters)
        else
            A[i, i] = 1.0 # b[i] = is already zero
        end
    end
    return b
end

# With this definition, `create_met_b!(A, mesh, conditions) = create_rhs_b!(A, mesh, conditions, Returns(-1), nothing)`.
# So, our problem can be defined by: 
using FiniteVolumeMethod, SparseArrays, DelaunayTriangulation, LinearSolve
const FVM = FiniteVolumeMethod
function poissons_equation(mesh::FVMGeometry,
    BCs::BoundaryConditions,
    ICs::InternalConditions=InternalConditions();
    diffusion_function,
    diffusion_parameters=nothing,
    source_function,
    source_parameters=nothing)
    conditions = Conditions(mesh, BCs, ICs)
    n = DelaunayTriangulation.num_solid_vertices(mesh.triangulation)
    A = zeros(n, n)
    FVM.triangle_contributions!(A, mesh, conditions, diffusion_function, diffusion_parameters)
    b = create_rhs_b!(A, mesh, conditions, source_function, source_parameters)
    return LinearProblem(sparse(A), b)
end

# Now let's test this problem. We consider 
# ```math 
# \begin{equation*}
# \begin{aligned}
# \grad^2 u &= -\sin(\pi x)\sin(\pi y) & \vb x \in [0,1]^2, \\
# u &= 0 & \vb x \in\partial[0,1]^2.
# \end{aligned}
# \end{equation*}
# ```
tri = triangulate_rectangle(0, 1, 0, 1, 100, 100, single_boundary=true)
mesh = FVMGeometry(tri)
BCs = BoundaryConditions(mesh, (x, y, t, u, p) -> zero(x), Dirichlet)
diffusion_function = (x, y, p) -> 1.0
source_function = (x, y, p) -> -sin(π * x) * sin(π * y)
prob = poissons_equation(mesh, BCs; diffusion_function, source_function)

#-
sol = solve(prob, KLUFactorization())

#-
using CairoMakie
fig, ax, sc = tricontourf(tri, sol.u, levels=LinRange(0, 0.05, 10), colormap=:matter, extendhigh=:auto)
tightlimits!(ax)
fig
using Test #src
@test sol.u ≈ [1 / (2π^2) * sin(π * x) * sin(π * y) for (x, y) in each_point(tri)] rtol = 1e-4 #src

# If we wanted to turn this into a `SteadyFVMProblem`, we use a similar call to `poissons_equation` 
# above except with an `initial_condition` for the initial guess. Moreover, we need to 
# change the sign of the source function, since above we are solving $\div[D(\vb x)\grad u] = f(\vb x)$,
# when `FVMProblem`s assume that we are solving $0 = \div[D(\vb x)\grad u] + f(\vb x)$.
initial_condition = zeros(num_points(tri))
fvm_prob = (SteadyFVMProblem ∘ FVMProblem)(mesh, BCs;
    diffusion_function=let D = diffusion_function
        (x, y, t, u, p) -> D(x, y, p)
    end,
    source_function=let S = source_function
        (x, y, t, u, p) -> -S(x, y, p)
    end,
    initial_condition,
    final_time=Inf)

#-
using SteadyStateDiffEq, OrdinaryDiffEq
fvm_sol = solve(fvm_prob, DynamicSS(TRBDF2(linsolve=KLUFactorization())))
using ReferenceTests #src
ax = Axis(fig[1, 2]) #src
tricontourf!(ax, tri, fvm_sol.u, levels=LinRange(0, 0.05, 10), colormap=:matter, extendhigh=:auto) #src
tightlimits!(ax) #src
resize_to_layout!(fig) #src
fig #src
@test_reference joinpath(@__DIR__, "../figures", "poissons_equation_template_1.png") fig #src
@test fvm_sol.u ≈ sol.u rtol = 1e-4 #src

# ## Using the Provided Template
# Let's now use the built-in `PoissonsEquation` which implements the above template 
# inside FiniteVolumeMethod.jl. The above problem can be constructed as follows:
prob = PoissonsEquation(mesh, BCs;
    diffusion_function=diffusion_function,
    source_function=source_function)

#-
sol = solve(prob, KLUFactorization())
@test sol.u ≈ [1 / (2π^2) * sin(π * x) * sin(π * y) for (x, y) in each_point(tri)] rtol = 1e-4 #src
@test sol.u ≈ fvm_sol.u rtol = 1e-4 #src

# Here is a benchmark comparison of the `PoissonsEquation` approach against the `FVMProblem` approach.
using BenchmarkTools
@btime solve($prob, $KLUFactorization());

#-
@btime solve($fvm_prob, $DynamicSS(TRBDF2(linsolve=KLUFactorization())));

# Let's now also solve a generalised Poisson equation. Following the example 
# in Section 7 of [this paper](https://my.ece.utah.edu/~ece6340/LECTURES/Feb1/Nagel%202012%20-%20Solving%20the%20Generalized%20Poisson%20Equation%20using%20FDM.pdf)
# by Nagel (2012), we consider an equation of the form 
# ```math 
# \div\left[\epsilon(\vb r)\grad V(\vb r)] = -\frac{\rho(\vb r)}{\epsilon_0}.
# ``` 
