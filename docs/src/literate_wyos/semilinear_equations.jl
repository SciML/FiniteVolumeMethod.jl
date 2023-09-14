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
# While this is no longer a linear problem, we can make use of a `SplitODEProblem` 
# from DifferentialEquations.jl which will treat the linear and nonlinear components separately.
# This allows, for example, implicit-explicit integrators to be used which will treat 
# $\vb A\vb u$ as the stiff component and then $\vb F(t, \vb u)$ as the non-stiff component, 
# or even algorithms like `LawsonEuler()` which directly exploits the linearity in $\vb A\vb u$.
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

# ## Implementation Details 
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

# Of course, in this case, `_neumann_boundary_edge_contributions` is not sufficient - 
# it needs to depend on `t` and `u`.