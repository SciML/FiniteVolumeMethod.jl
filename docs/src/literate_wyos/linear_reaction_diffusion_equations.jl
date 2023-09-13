# # Linear Reaction-Diffusion Equations 
# ```@contents
# Pages = ["linear_reaction_diffusion_equations.md"]
# ```
# Next, we write a specialised solver for solving linear reaction-diffusion equations. What 
# we produce in this section can also be accessed in `FiniteVolumeMethod.LinearReactionDiffusionEquation`.

# ## Mathematical Details 
# To start, let's give the mathematical details. The problems we will be solving take the form 
# ```math 
# \pdv{u}{t} = \div\left[D(\vb x)\grad u\right] + f(\vb x)u.
# ```
# We want to turn this into an equation of the form $\mathrm d\vb u/\mathrm dt = \vb A\vb u + \vb b$ 
# as usual. This takes the same form as our [diffusion equation example](diffusion_equations.md),
# except with the extra $f(\vb x)u$ term, which just adds an exta $f(\vb x)$ term 
# to the diagonal of $\vb A$. See the previois sections for further mathematical details. 

# ## Implementation 
# Let us now implement the solver. For constructing $\vb A$, we can use `FiniteVolumeMethod.triangle_contributions!`
# as in the previous sections, but we will need an extra function to add $f(\vb x)$ to the appropriate diagonals.
# We can also reuse `apply_dirichlet_conditions!`, `apply_dudt_conditions`, and 
# `boundary_edge_contributions!` from the diffusion equation example. Here is our implementation. 
using FiniteVolumeMethod, SparseArrays, OrdinaryDiffEq, LinearAlgebra
const FVM = FiniteVolumeMethod
function linear_source_contributions!(A, mesh, conditions, source_function, source_parameters)
    for i in each_solid_vertex(mesh.triangulation)
        if !FVM.has_condition(conditions, i)
            x, y = get_point(mesh, i)
            A[i, i] += source_function(x, y, source_parameters)
        end
    end
end
function linear_reaction_diffusion_equation(mesh::FVMGeometry,
    BCs::BoundaryConditions,
    ICs::InternalConditions=InternalConditions();
    diffusion_function,
    diffusion_parameters=nothing,
    source_function,
    source_parameters=nothing,
    initial_condition,
    initial_time=0.0,
    final_time)
    conditions = Conditions(mesh, BCs, ICs)
    n = DelaunayTriangulation.num_solid_vertices(mesh.triangulation)
    Afull = zeros(n + 1, n + 1)
    A = @views Afull[begin:end-1, begin:end-1]
    b = @views Afull[begin:end-1, end]
    _ic = vcat(initial_condition, 1)
    FVM.triangle_contributions!(A, mesh, conditions, diffusion_function, diffusion_parameters)
    FVM.boundary_edge_contributions!(A, b, mesh, conditions, diffusion_function, diffusion_parameters)
    linear_source_contributions!(A, mesh, conditions, source_function, source_parameters)
    FVM.apply_dudt_conditions!(b, mesh, conditions)
    FVM.apply_dirichlet_conditions!(_ic, mesh, conditions)
    Af = sparse(Afull)
    prob = ODEProblem(MatrixOperator(Af), _ic, (initial_time, final_time))
    return prob
end

# If you go and look back at the `diffusion_equation` function from the 
# [diffusion equation example](diffusion_equations.md), you will see that
# this is essentially the same function except we now have `linear_source_contributions!`
# and `source_function` and `source_parameters` arguments.

# Let's now test this function. We consider the problem
# ```math 
# \pdv{T}{t} = \div\left[10^{-3}x^2y\grad T\right] + (x-1)(y-1)T, \quad \vb x \in [0,1]^2,
# ```
# with $\grad T \vdot\vu n = 1$.
using DelaunayTriangulation
tri = triangulate_rectangle(0, 1, 0, 1, 150, 150, single_boundary=true)
mesh = FVMGeometry(tri)
BCs = BoundaryConditions(mesh, (x, y, t, u, p) -> one(x), Neumann)
diffusion_function = (x, y, p) -> p.D * x^2 * y
diffusion_parameters = (D=1e-3,)
source_function = (x, y, p) -> (x - 1) * (y - 1)
initial_condition = [x^2 + y^2 for (x, y) in each_point(tri)]
final_time = 8.0
prob = linear_reaction_diffusion_equation(mesh, BCs;
    diffusion_function, diffusion_parameters,
    source_function, initial_condition, final_time)

#-
sol = solve(prob, Tsit5(); saveat=2)

#-
using CairoMakie
fig = Figure(fontsize=38)
for j in eachindex(sol)
    ax = Axis(fig[1, j], width=600, height=600,
        xlabel="x", ylabel="y",
        title="t = $(sol.t[j])")
    tricontourf!(ax, tri, sol.u[j], levels=0:0.1:1, extendlow=:auto, extendhigh=:auto, colormap=:turbo)
    tightlimits!(ax)
end
resize_to_layout!(fig)
fig

# Here is how we could convert this into an `FVMProblem`. Note that the Neumann 
# boundary conditions are expressed as $\grad T\vdot\vu n = 1$ above, but for `FVMProblem` 
# we need them in the form $\vb q\vdot\vu n = \ldots$. For this problem, $\vb q=-D\grad T$,
# which gives $\vb q\vdot\vu n = -D$.
_BCs = BoundaryConditions(mesh, (x, y, t, u, p) -> -p.D(x, y, p.Dp), Neumann;
    parameters=(D=diffusion_function, Dp=diffusion_parameters))
fvm_prob = FVMProblem(
    mesh,
    _BCs;
    diffusion_function=let D=diffusion_function 
        (x, y, t, u, p) -> D(x, y, p)
    end,
    diffusion_parameters=diffusion_parameters,
    source_function=let S=source_function 
        (x, y, t, u, p) -> S(x, y, p) * u
    end,
    final_time=final_time,
    initial_condition=initial_condition
)
using LinearSolve
fvm_sol = solve(fvm_prob, TRBDF2(linsolve=KLUFactorization()), saveat=2.0)

for j in eachindex(fvm_sol) #src
    ax = Axis(fig[2, j], width=600, height=600, #src
        xlabel="x", ylabel="y", #src
        title="t = $(fvm_sol.t[j])") #src
    tricontourf!(ax, tri, fvm_sol.u[j], levels=0:0.1:1, extendlow=:auto, extendhigh=:auto, colormap=:turbo) #src
    tightlimits!(ax) #src
end #src
resize_to_layout!(fig) #src
using ReferenceTests #src
@test_reference joinpath(@__DIR__, "../figures", "linear_reaction_diffusion_equation_template_1.png") fig #src

# ## Using the Provided Template 
# The above code is implemented in `LinearReactionDiffusionEquation` in FiniteVolumeMethod.jl. 
prob = LinearReactionDiffusionEquation(mesh, BCs;
    diffusion_function, diffusion_parameters,
    source_function,  initial_condition, final_time)
sol = solve(prob, Tsit5(); saveat=2)

using Test #src
@test sol[begin:end-1, 2:end] ≈ fvm_sol[:, 2:end] rtol=1e-1 #src

# Here is a benchmark comparison of `LinearReactionDiffusionEquation` versus `FVMProblem`.
using BenchmarkTools 
@btime solve($prob, $Tsit5(); saveat=$2);

#-
@btime solve($fvm_prob, $TRBDF2(linsolve=$KLUFactorization()); saveat=$2);