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
# in [the diffusion equation section](diffusion_equations.md).