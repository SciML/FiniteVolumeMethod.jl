# # Semilinear Equations 
# ```@contents 
# Pages = ["semilinear_equations.md"]
# ``` 
# Now we consider semilinear equations, where the diffusion term is linear and the source term is nonlinear. What 
# we produce in this section also be accessed in `FiniteVolumeMethod.SemilinearEquations`.

# ## Mathematical Details 
# We start by giving the mathematical details. We are considering problems of the form 
# ```math 
# \pdv{u}{t} = \div\left[D(\vb x)\grad u\right] + f(\vb x, t, u).
# ```
# This is similar to the linear reaction-diffusion equations developed [previously](linear_reaction_diffusion_equations.md),
# except now the source term is nonlinear (previously, it was $f(\vb x)u$), which is much more common. The 
# mathematical details do not significantly change, though - the main change is the implementation later. 
#
# We know that 
# ```math 
# \begin{equation*}
# \begin{aligned}
# \dv{u_i}{t} &= \frac{1}{V_i}\sum_{\sigma\in\mathcal E_i} D(\vb x_\sigma)\left[\left(s_{k, 11}n_\sigma^x + s_{k, 21}n_\sigma^y\right)u_{k1} + \left(s_{k, 12}n_\sigma^x + s_{k, 22}n_\sigma^y\right)u_{k2} + \left(s_{k, 13}n_\sigma^x + s_{k, 23}n_\sigma^y\right)u_{k3}\right]L_\sigma \\&+ f(\vb x_i, t, u_i).
# \end{aligned}
# \end{equation*}
# ```
# Thus, modulo some boundary condition details, 
# ```math 
# \dv{\vb u}{t} = \vb A\vb u + \vb F(t, \vb u), \quad \vb F(t, \vb u) = \begin{bmatrix} f(\vb x_1,t,\vb u_1) \\ \vdots \\ f(\vb x_n, t, \vb u_n) \end{bmatrix}.
# ```
# While this is no longer a linear problem, being able to quickly evaluate $\vb A\vb u$ is 
# extremely useful. In particular, we can represent this problem as a `SplitODEProblem`.
#
# Let us now think about the boundary condition details. For the Neumann boundary conditions,
# we need to think about what happens at a Neumann edge. When we are building $\vb A$ and we encounter 
# such an edge, we put the contribution from that edge into another vector $\vb b$. For the 
# [`DiffusionEquation`](diffusion_equations.md) template, we placed this $\vb b$ inside of $\vb A$,
# redefining
# ```math 
# \tilde{\vb A} = \begin{bmatrix} \vb A & \vb b \\ \vb 0^{\mkern-1.5mu\mathsf T} & 0 \end{bmatrix},
# ```
# and so we could do that here as well. Alternatively, if we kept $\vb b$ separate,
# then we could allow for inhomogeneous Neumann boundary conditions, putting $\vb b + \vb F(t, \vb u)$
# into the nonlinear component. To make this template as generic as we can, let us allow for this latter case, 
# pushing $\vb b$ into the nonlinear component. This does mean that more time might be spent than is necessary 
# for boundary conditions, but that is not where the primary costs come from anyway.
#
# For handling Dirichlet boundary conditions, to allow them to depend on time and on $u$, we just use 
# a `DiscreteCallback` from DifferentialEquations.jl. This is the same as what we do for `FVMProblems`.
# For `Dudt` conditions, these also get placed into $\vb F(t, \vb u)$, provided we are careful to 
# make the corresponding row of $\vb A$ all zero.

# ## Implementation
# Now that we understand the structure of the problem, let's implement it.
# For the first part of the implementation, we need 
# the function that computes $\vb A$. We already have the function 
# that does this for us - it's what we used for building `DiffusionEquation`
# in [the diffusion equation section](diffusion_equations.md). We do need to modify 
# it slightly so that Neumannn edges are avoided, instead considering them only 
# in the evaluation of $\vb F$.
#
# To start, let's rewrite `FiniteVolumeMethod.boundary_edge_contributions!` to reuse 
# a smaller function, allowing us to more easily split it into the case of evaluating non-Neumann 
# edges versus Neumann edges. We have put this into `FiniteVolumeMethod.non_neumann_boundary_edge_contributions!`
# and `FiniteVolumeMethod.neumann_boundary_edge_contributions!`, which we redefine below for pedagogical purposes.
using FiniteVolumeMethod, DelaunayTriangulation
const FVM = FiniteVolumeMethod
function _boundary_edge_contributions!(A, b, mesh, conditions,
    diffusion_function, diffusion_parameters)
    _non_neumann_boundary_edge_contributions!(A, mesh, conditions, diffusion_function, diffusion_parameters)
    _neumann_boundary_edge_contributions!(b, mesh, conditions, diffusion_function, diffusion_parameters)
    return nothing
end
function _neumann_boundary_edge_contributions!(b, mesh, conditions, diffusion_function, diffusion_parameters)
    for (e, fidx) in FVM.get_neumann_edges(conditions)
        i, j = DelaunayTriangulation.edge_indices(e)
        _, _, mᵢx, mᵢy, mⱼx, mⱼy, ℓ, _, _ = FVM.get_boundary_cv_components(mesh, i, j)
        Dᵢ = diffusion_function(mᵢx, mᵢy, diffusion_parameters)
        Dⱼ = diffusion_function(mⱼx, mⱼy, diffusion_parameters)
        i_hascond = FVM.has_condition(conditions, i)
        j_hascond = FVM.has_condition(conditions, j)
        aᵢ = FVM.eval_condition_fnc(conditions, fidx, mᵢx, mᵢy, nothing, nothing)
        aⱼ = FVM.eval_condition_fnc(conditions, fidx, mⱼx, mⱼy, nothing, nothing)
        i_hascond || (b[i] += Dᵢ * aᵢ * ℓ / FVM.get_volume(mesh, i))
        j_hascond || (b[j] += Dⱼ * aⱼ * ℓ / FVM.get_volume(mesh, j))
    end
    return nothing
end
function _non_neumann_boundary_edge_contributions!(A, mesh, conditions, diffusion_function, diffusion_parameters)
    for e in keys(get_boundary_edge_map(mesh.triangulation))
        i, j = DelaunayTriangulation.edge_indices(e)
        if !FVM.is_neumann_edge(conditions, i, j)
            nx, ny, mᵢx, mᵢy, mⱼx, mⱼy, ℓ, T, props = FVM.get_boundary_cv_components(mesh, i, j)
            ijk = indices(T)
            s₁₁, s₁₂, s₁₃, s₂₁, s₂₂, s₂₃, s₃₁, s₃₂, s₃₃ = props.shape_function_coefficients
            Dᵢ = diffusion_function(mᵢx, mᵢy, diffusion_parameters)
            Dⱼ = diffusion_function(mⱼx, mⱼy, diffusion_parameters)
            i_hascond = FVM.has_condition(conditions, i)
            j_hascond = FVM.has_condition(conditions, j)
            aᵢ123 = (Dᵢ * ℓ * (s₁₁ * nx + s₂₁ * ny),
                Dᵢ * ℓ * (s₁₂ * nx + s₂₂ * ny),
                Dᵢ * ℓ * (s₁₃ * nx + s₂₃ * ny))
            aⱼ123 = (Dⱼ * ℓ * (s₁₁ * nx + s₂₁ * ny),
                Dⱼ * ℓ * (s₁₂ * nx + s₂₂ * ny),
                Dⱼ * ℓ * (s₁₃ * nx + s₂₃ * ny))
            for vert in 1:3
                i_hascond || (A[i, ijk[vert]] += aᵢ123[vert] / FVM.get_volume(mesh, i))
                j_hascond || (A[j, ijk[vert]] += aⱼ123[vert] / FVM.get_volume(mesh, i))
            end
        end
    end
    return nothing
end

# Of course, in this case, `_neumann_boundary_edge_contributions!` is not sufficient - 
# it needs to depend on `t` and `u`. Before we add on this dependence, let us think
# about what values we actually need. For a given Neumann edge $e_{ij}$ with midpoint 
# \vb x_\sigma$, we evaluate $u$ at $\vb x_{i'} = (\vb x_i+\vb x_\sigma)/2$ 
# and at $\vb x_{j'} = (\vb x_\sigma + \vb x_j)/2$. In the main finite volume code, 
# $u$ gets replaced with the linear interpolant $\alpha x + \beta y + \gamma$ inside 
# the triangle adjoining this boundary edge. When this linear interpolant is evaluated 
# along the boundary edges, all we need is two-point interpolation. In particular,
# to evaluate $u$ at a point $\vb x \in e_{ij}$, we can use 
# ```math 
# u(\vb x, t) \approx u_i(t) + \frac{u_j(t) - u_i(t)}{\|\vb x_j-\vb x_i\|}\|\vb x - \vb x_i\|,
# ```
# To check that this formula is correct, note
# ```math 
# \begin{equation*}
# \begin{aligned}
# u(\vb x_i, t) &\approx u_i(t) + \frac{u_j(t) - u_i(t)}{\|\vb x_j-\vb x_i\|}\|\vb x_i - \vb x_i\| = u_i(t), \\
# u(\vb x_j, t) &\approx u_i(t) + \frac{u_j(t) - u_i(t)}{\|\vb x_j-\vb x_i\|}\|\vb x_j - \vb x_i\| \\
# &= u_i(t) + (u_j(t) - u_i(t)) = u_j(t).
# \end{aligned}
# \end{equation*}
# ```
# Let's now apply this to write `_neumann_boundary_edge_contributions!` in a way that allows for 
# `u` and `t` to be used. For this new function, the interpretation of `b` is now in terms of $\vb F$ above.
function two_point_interpolant(mesh, u::AbstractVector, i, j, mx, my)
    xᵢ, yᵢ = get_point(mesh, i)
    xⱼ, yⱼ = get_point(mesh, j)
    ℓ = sqrt((xⱼ - xᵢ)^2 + (yⱼ - yᵢ)^2)
    ℓ′ = sqrt((mx - xᵢ)^2 + (my - yᵢ)^2)
    return u[i] + (u[j] - u[i]) * ℓ′ / ℓ
end
function _neumann_boundary_edge_contributions!(F, mesh, conditions, diffusion_function, diffusion_parameters, u, t)
    for (e, fidx) in FVM.get_neumann_edges(conditions)
        i, j = DelaunayTriangulation.edge_indices(e)
        _, _, mᵢx, mᵢy, mⱼx, mⱼy, ℓ, _, _ = FVM.get_boundary_cv_components(mesh, i, j)
        Dᵢ = diffusion_function(mᵢx, mᵢy, diffusion_parameters)
        Dⱼ = diffusion_function(mⱼx, mⱼy, diffusion_parameters)
        i_hascond = FVM.has_condition(conditions, i)
        j_hascond = FVM.has_condition(conditions, j)
        uᵢ_itp = two_point_interpolant(mesh, u, i, j, mᵢx, mᵢy)
        uⱼ_itp = two_point_interpolant(mesh, u, i, j, mⱼx, mⱼy)
        aᵢ = FVM.eval_condition_fnc(conditions, fidx, mᵢx, mᵢy, t, uᵢ_itp)
        aⱼ = FVM.eval_condition_fnc(conditions, fidx, mⱼx, mⱼy, t, uⱼ_itp)
        i_hascond || (F[i] += Dᵢ * aᵢ * ℓ / FVM.get_volume(mesh, i))
        j_hascond || (F[j] += Dⱼ * aⱼ * ℓ / FVM.get_volume(mesh, j))
    end
    return nothing
end

# Now we can write the function that constructs $\vb A$.
using SparseArrays
function get_semilinear_matrix(mesh, conditions, diffusion_function, diffusion_parameters)
    n = DelaunayTriangulation.num_solid_vertices(mesh.triangulation)
    A = zeros(n, n)
    FVM.triangle_contributions!(A, mesh, conditions, diffusion_function, diffusion_parameters)
    _non_neumann_boundary_edge_contributions!(A, mesh, conditions, diffusion_function, diffusion_parameters)
    return sparse(A)
end

# Now we need the function that will evaluate the source term. This function needs to take the form `(du, u, p, t)`,
# where `du` is the $\vb F$ from above. Here is the definition. 
function eval_semilinear_source!(du, u, p, t)
    fill!(du, zero(eltype(du)))
    A, mesh, conditions, diffusion_function, diffusion_parameters, source_function, source_parameters = p
    _neumann_boundary_edge_contributions!(du, mesh, conditions, diffusion_function, diffusion_parameters, u, t)
    for i in each_solid_vertex(mesh.triangulation)
        if !FVM.has_condition(conditions, i)
            x, y = get_point(mesh, i)
            du[i] += source_function(x, y, t, u[i], source_parameters)
        end
    end
    FVM.apply_dudt_conditions!(du, mesh, conditions)
    return du
end

# To finish the problem, we need the Dirichlet callback. The function that 
# does this for an `FVMProblem` is `update_dirichlet_nodes!`, but we need to make some 
# changes for this to work, since `update_dirichlet_nodes!` was designed specifically for `AbstractFVMProblem`s.
# It's easier to just write a new method.
function semilinear_equation_update_dirichlet_nodes!(integrator)
    mesh, conditions = integrator.p
    for (i, function_index) in FVM.get_dirichlet_nodes(conditions)
        p = get_point(mesh, i)
        x, y = getxy(p)
        integrator.u[i] = FVM.eval_condition_fnc(conditions, function_index, x, y, integrator.t, integrator.u[i])
    end
    return nothing
end

using LinearAlgebra
function _semil_f!(du, u, p, t)
    mul!(du, p.A, u)
    eval_semilinear_source!(du, u, p, t)
end

# We will use `get_dirichlet_callback` to construct this callback. Now we can actually define the problem.
function semilinear_equation(mesh::FVMGeometry,
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
    A = get_semilinear_matrix(mesh, conditions, diffusion_function, diffusion_parameters)
    A_op = MatrixOperator(A)
    f = SplitFunction(A_op, eval_semilinear_source!)
    #f = ODEFunction{true}(_semil_f!; jac_prototype=FVM.jacobian_sparsity(mesh.triangulation))
    p = (A=A,mesh, conditions, diffusion_function, diffusion_parameters, source_function, source_parameters)
    cb = FVM.get_dirichlet_callback(conditions, semilinear_equation_update_dirichlet_nodes!)
    prob = SplitODEProblem(f, initial_condition, (initial_time, final_time), p, callback=cb)
    return prob
end

# Now let's test this problem, using:
# ```math 
# \begin{equation*}
# \begin{aligned}
# \pdv{u}{t} &= \grad^2 u + ru\left(1-\frac{u}{K}), & \vb x \in \Omega, \\
# \grad u \vdot \vu n &= 1, & \vb x \in \partial\Omega,
# \end{aligned}
# \end{equation*}
# ```
# where $\Omega = [0, 100]^2$, $r = 1$, and $K=1$. The initial condition will be $u(\vb x, 0) = 0$, except 
# $u(50, 50, 0) = 1$.
using OrdinaryDiffEq, LinearSolve
tri = triangulate_rectangle(0, 100, 0, 100, 121, 121, single_boundary=true)
mid_idx = findfirst(==((50,50)), each_point(tri))
mesh = FVMGeometry(tri)
BCs = BoundaryConditions(mesh, (x, y, t, u, p) -> one(u), Neumann)
diffusion_function = (x, y, p) -> 1.0
r = 1.0
K = 1.0
source_function = (x, y, t, u, p) -> p.r * u * (1 - u / p.K)
source_parameters = (r=r, K=K)
initial_condition = zeros(num_points(tri))
initial_condition[mid_idx] = 1.0
prob = semilinear_equation(mesh, BCs;
    diffusion_function=diffusion_function,
    source_function=source_function,
    source_parameters=source_parameters,
    initial_condition,
    final_time=20.0)

#-
