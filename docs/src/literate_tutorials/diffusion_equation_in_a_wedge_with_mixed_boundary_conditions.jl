using DisplayAs #hide
tc = DisplayAs.withcontext(:displaysize => (15, 80), :limit => true); #hide
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
# \end{equation*}
# ```
# where we take $f(r,\theta) = 1-r$ and $\alpha=\pi/4$.
#
# Note that the PDE is provided in polar form, but Cartesian coordinates 
# are assumed for the operators in our code. The conversion is easy, noting 
# that the two Neumann conditions are just equations of the form $\grad u \vdot \vu n = 0$.
# Moreover, although the right-hand side of the PDE is given as a Laplacian, 
# recall that $\grad^2 = \div\grad$, so we can write the PDE as $\partial u/\partial t + \div \vb q = 0$,
# where $\vb q = -\grad u$.
#
# Let us now setup the problem. To define the geometry, 
# we need to be careful that the `Triangulation` recognises 
# that we need to split the boundary into three parts,
# one part for each boundary condition. This is accomplished 
# by providing a single vector for each part of the boundary as follows
# (and as described in DelaunayTriangulation.jl's documentation),
# where we also `refine!` the mesh to get a better mesh. For the arc, 
# we use the `CircularArc` so that the mesh knows that it is triangulating 
# a certain arc in that area.
using DelaunayTriangulation, FiniteVolumeMethod, ElasticArrays
using ReferenceTests, Bessels, FastGaussQuadrature, Cubature #src

α = π / 4
points = [(0.0, 0.0), (1.0, 0.0), (cos(α), sin(α))]
bottom_edge = [1, 2]
arc = CircularArc((1.0, 0.0), (cos(α), sin(α)), (0.0, 0.0))
upper_edge = [3, 1]
boundary_nodes = [bottom_edge, [arc], upper_edge]
tri = triangulate(points; boundary_nodes)
A = get_area(tri)
refine!(tri; max_area=1e-4A)
mesh = FVMGeometry(tri)

# This is the mesh we've constructed.
using CairoMakie
fig, ax, sc = triplot(tri)
fig

# To confirm that the boundary is now in three parts, see:
get_boundary_nodes(tri)

# We now need to define the boundary conditions. For this, 
# we need to provide `Tuple`s, where the `i`th element of the 
# `Tuple`s refers to the `i`th part of the boundary. The boundary 
# conditions are thus:
lower_bc = arc_bc = upper_bc = (x, y, t, u, p) -> zero(u)
types = (Neumann, Dirichlet, Neumann)
BCs = BoundaryConditions(mesh, (lower_bc, arc_bc, upper_bc), types)

# Now we can define the PDE. We use the reaction-diffusion formulation, 
# specifying the diffusion function as a constant. 
f = (x, y) -> 1 - sqrt(x^2 + y^2)
D = (x, y, t, u, p) -> one(u)
initial_condition = [f(x, y) for (x, y) in DelaunayTriangulation.each_point(tri)]
final_time = 0.1
prob = FVMProblem(mesh, BCs; diffusion_function=D, initial_condition, final_time)

# If you did want to use the flux formulation, you would need to provide 
flux = (x, y, t, α, β, γ, p) -> (-α, -β)

# which replaces `u` with `αx + βy + γ` so that we approximate $\grad u$ by $(\alpha,\beta)^{\mkern-1.5mu\mathsf{T}}$,
# and the negative is needed since $\vb q = -\grad u$.

# We now solve the problem. We provide the solver for this problem.
# In my experience, I've found that `TRBDF2(linsolve=KLUFactorization())` typically 
# has the best performance for these problems.
using OrdinaryDiffEq, LinearSolve
sol = solve(prob, TRBDF2(linsolve=KLUFactorization()), saveat=0.01, parallel=Val(false))
ind = findall(DelaunayTriangulation.each_point_index(tri)) do i #hide
    !DelaunayTriangulation.has_vertex(tri, i) #hide
end #hide
@test sol[ind, :] ≈ reshape(repeat(initial_condition, length(sol)), :, length(sol))[ind, :] # make sure that missing vertices don't change #hide
sol |> tc #hide

#-
using CairoMakie
fig = Figure(fontsize=38)
for (i, j) in zip(1:3, (1, 6, 11))
    ax = Axis(fig[1, i], width=600, height=600,
        xlabel="x", ylabel="y",
        title="t = $(sol.t[j])",
        titlealign=:left)
    tricontourf!(ax, tri, sol.u[j], levels=0:0.01:1, colormap=:matter)
    tightlimits!(ax)
end
resize_to_layout!(fig)
fig
@test_reference joinpath(@__DIR__, "../figures", "diffusion_equation_in_a_wedge_with_mixed_boundary_conditions.png") fig #src

function get_ζ_terms(M, N, α) #src
    ζ = zeros(M, N + 2) #src
    for n in 0:(N+1) #src
        order = n * π / α #src
        @views ζ[:, n+1] .= approx_besselroots(order, M) #src
    end #src
    return ζ #src
end #src
function get_sum_coefficients(M, N, α, ζ) #src
    _f = (r, θ) -> 1.0 - r #src
    A = zeros(M, N + 1) # A[m, n+1] is the coefficient Aₙₘ #src
    for n in 0:N #src
        order = n * π / α #src
        for m in 1:M #src
            integrand = rθ -> _f(rθ[2], rθ[1]) * besselj(order, ζ[m, n+1] * rθ[2]) * cos(order * rθ[1]) * rθ[2] #src
            A[m, n+1] = 4.0 / (α * besselj(order + 1, ζ[m, n+1])^2) * hcubature(integrand, [0.0, 0.0], [α, 1.0]; abstol=1e-8)[1] #src
        end #src
    end #src
    return A #src
end #src
function exact_solution(x, y, t, A, ζ, f, α) #src
    t == 0.0 && return f(x, y) #src
    r = sqrt(x^2 + y^2) #src
    θ = atan(y, x) #src
    u, v = size(ζ) #src
    M, N = u, v - 2 #src
    s = 0.0 #src
    for m in 1:M #src
        s += 0.5 * A[m, 1] * exp(-ζ[m, 1]^2 * t) * besselj(0.0, ζ[m, 1] * r) #src
    end #src
    for n in 1:N #src
        order = n * π / α #src
        for m = 1:M #src
            s += +A[m, n+1] * exp(-ζ[m, n+1]^2 * t) * besselj(order, ζ[m, n+1] * r) * cos(order * θ) #src
        end #src
    end #src
    return s #src
end #src
function compare_solutions(sol, tri, α, f) #src
    n = DelaunayTriangulation.num_points(tri) #src
    x = zeros(n, length(sol)) #src
    y = zeros(n, length(sol)) #src
    u = zeros(n, length(sol)) #src
    ζ = get_ζ_terms(20, 20, α) #src
    A = get_sum_coefficients(20, 20, α, ζ) #src
    for i in eachindex(sol) #src
        for j in each_solid_vertex(tri) #src
            x[j, i], y[j, i] = get_point(tri, j) #src
            u[j, i] = exact_solution(x[j, i], y[j, i], sol.t[i], A, ζ, f, α) #src
        end #src
    end #src
    return x, y, u #src
end #src
x, y, u = compare_solutions(sol, tri, α, f) #src
fig = Figure(fontsize=64) #src
for i in eachindex(sol) #src
    ax = Axis(fig[1, i], width=600, height=600) #src
    tricontourf!(ax, tri, sol.u[i], levels=0:0.01:1, colormap=:matter) #src
    ax = Axis(fig[2, i], width=600, height=600) #src
    tricontourf!(ax, tri, u[:, i], levels=0:0.01:1, colormap=:matter) #src
end #src
resize_to_layout!(fig) #src
fig #src
@test_reference joinpath(@__DIR__, "../figures", "diffusion_equation_in_a_wedge_with_mixed_boundary_conditions_exact_comparisons.png") fig #src