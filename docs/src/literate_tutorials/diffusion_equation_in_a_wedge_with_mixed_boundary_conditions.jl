# # Diffusion Equation in a Wedge with Mixed Boundary Conditions 
# In this example, we consider a diffusion equation on a wedge 
# with angle $\alpha$ and mixed boundary conditions:
# ```math
# \begin{equation*}
# \begin{aligned}
# \pdv{u(r, \theta, t)}{t} &= \grad^2u(r,\theta,t), & 0<r<1,\,0<\theta<\alpha,\,t>0,\\[6pt]
# \pdv{u(r, 0, t)}{\theta} & = 0 & 0<r<1,\,t>0,\\[6pt]
# u(1, \theta, t) &= 0 & 0<\theta<\alpha,\,t>0,\\[6pt]
# \pdv{u(r,\alpha,t)}{\theta} & = 0 & 0<\theta<\alpha,\,t>0,\\[6pt]
# u(r, \theta, 0) &= f(r,\theta) & 0<r<1,\,0<\theta<\alpha,
# \end{aligned}
# ```
# where we take $f(r,\theta) = 1-r$ and $\alpha=\pi/4$.
#
# Note that the PDE is provided in polar form, but Cartesian coordinates 
# are assumed for the operators in our code. The conversion is easy, noting 
# that the two Neumann conditions are just equations of the form $\grad u \vdot \hat{\vu n} = 0$.
# Moreover, although the right-hand side of the PDE is given as a Laplacian, 
# recall that $\grad^2 = \div\grad$, so we can write the PDE as $\partial u/\partial t + \div q = 0$,
# where $\div q = -\grad u$.
#
# Let us now setup the problem. To define the geometry, 
# we need to be careful that the `Triangulation` recognises 
# that we need to split the boundary into three parts,
# one part for each boundary condition. This is accomplished 
# by providing a single vector for each part of the boundary as follows
# (and as described in DelaunayTriangulation.jl's documentation),
# where we also `refine!` the mesh to get a better mesh. 
using DelaunayTriangulation, FiniteVolumeMethod, ElasticArrays
n = 50
α = π / 4
# The bottom edge 
x₁ = [0.0, 1.0]
y₁ = [0.0, 0.0]
# The arc 
r₂ = fill(1, n)
θ₂ = LinRange(0, α, n)
x₂ = @. r₂ * cos(θ₂)
y₂ = @. r₂ * sin(θ₂)
# The upper edge 
x₃ = [cos(α), 0.0]
y₃ = [sin(α), 0.0]
# Now combine and create the mesh 
x = [x₁, x₂, x₃]
y = [y₁, y₂, y₃]
boundary_nodes, points = convert_boundary_points_to_indices(x, y; existing_points=ElasticMatrix{Float64}(undef, 2, 0))
tri = triangulate(points; boundary_nodes)
A = get_total_area(tri)
refine!(tri; max_area=1e-4A)
mesh = FVMGeometry(tri)

# This is the mesh we've constructed.
fig, ax, sc = triplot(tri)
fig

# To confirm that the boundary is now in three parts, see:
get_boundary_nodes(tri)

# We now need to define the boundary conditions. For this, 
# we need to provide `Tuple`s, where the `i`th element of the 
# `Tuple`s refers to the `i`th part of the boundary. The boundary 
# conditions are thus:
lower_bc = arc_bc=upper_bc=(x,y,t,u,p)->zero(u)
types = (Neumann, Dirichlet, Neumann)
BCs = BoundaryConditions(mesh, (lower_bc, arc_bc, upper_bc), types)

# Now we can define the PDE. We use the reaction-diffusion formulation, 
# specifying the diffusion function as a constant. 
f = (x, y) -> 1 - sqrt(x^2 + y^2)
D = (x, y, t, u, p) -> one(u)
initial_condition = [f(x, y) for (x, y) in each_point(tri)]
final_time = 0.1
prob = FVMProblem(mesh, BCs; diffusion_function=D, initial_condition, final_time)

# If you did want to use the flux formulation, you would need to provide 
flux = (x, y, t, α, β, γ, p) -> (-α, -β)

# which replace `u` with `αx + βy + γ` so that we approximate $\grad u$ by $(\alpha,\beta)^{\mkern-1.5mu\mathsf{T}}$,
# and the negative is needed since $\div q = -\grad u$.

# We now solve the problem. We provide the solver for this problem.
# In my experience, I've found that `TRBDF2(linsolve=KLUFactorization())` typically 
# has the best performance for these problems.
using OrdinaryDiffEq, LinearSolve
sol = solve(prob, TRBDF2(linsolve=KLUFactorization()), saveat=0.01)
