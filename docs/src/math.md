# Mathematical and Implementation Details

Here we outline the mathematical and implementation details involved in implementing the finite volume method (FVM). We assume that our partial differential equation (PDE) is given by 
```math
\begin{equation}
\label{eq:pde}
\pdv{u(\vb x, t)}{t} + \div \vb q(\vb x, t) = S(\vb x, t), \quad \vb x \in \Omega,
\end{equation}
```
together with some boundary conditions or internal conditions that we discuss later.

# Interior Discretisation 

Let us start by discretising \eqref{eq:pde} inside $\Omega$. The first step in this discretisation is to compute a triangulation of $\Omega$, decomposing $\Omega$ into a collection of disjoint triangles $\{T_k\}$ so that 
```math
\Omega = \bigcup_k T_k.
```
This triangulation is typically a _constrained Delaunay triangulation_, denoted $\mathcal T(\Omega)$, with appropriate care taken if $\Omega$ is multiply-connected; these triangulations can be computed using [DelaunayTriangulation.jl](https://github.com/DanielVandH/DelaunayTriangulation.jl). An example of such a domain $\Omega$ and its triangulation $\mathcal T(\Omega)$ is shown below, where we use a multiply-connected domain to emphasise that these details are not necessarily restricted to simple domains.

```@setup ex_tri
using DelaunayTriangulation, CairoMakie
θ = LinRange(0, 2π, 50)
θ = collect(θ)
θ[end] = θ[begin]
g = θ -> cos(θ) - sin(3θ) + sin(5θ)
ε = 1 / 10
x = @. (1 + g(θ) * ε) * cos(θ)
y = @. (1 + g(θ) * ε) * sin(θ)
h = θ -> cos(3θ) - sin(θ)
ε = 1 / 5
x2 = @. 0.5(1 + h(θ) * ε) * cos(θ)
y2 = @. 0.5(1 + h(θ) * ε) * sin(θ)
reverse!(x2)
reverse!(y2)
boundary_nodes, points = convert_boundary_points_to_indices([[x], [x2]], [[y], [y2]])
tri = triangulate(points; boundary_nodes)
refine!(tri)
fig, ax, sc = triplot(tri,
    show_convex_hull=false,
    show_constrained_edges=false,
    show_all_points=true,
    point_color=:black)
hidedecorations!(ax)
```

```@example ex_tri
fig #hide
```

## Control volumes

Key to the FVM are the _control volumes_, which are used to define volumes $\Omega_i$ around individual vertices $\vb x_i$ that we integrate the PDE over. To define these volumes, take $\vb x_i \in \Omega$, which is a vertex of $\mathcal T(\Omega)$, and take the set of triangles $\mathcal T_i = \{T_k\}$ that have $\vb x_i$ as a vertex. For each of these triangles $T_{k'} \in \mathcal T_i$, connect its centroid to the midpoints of the triangle's edges. Once this procedure is complete, we obtain a closed polygon around $\vb x_i$ that we denote by $\partial\Omega_i$ and whose interior defines the control volume $\Omega_i$. We show the result of this procedure, applied to the above domain, below, where we show the centroids in red and the control volume boundaries in blue.

```@setup ex_tri 
centroids = NTuple{2,Float64}[]
linesegments = NTuple{2,Float64}[]
for T in each_solid_triangle(tri)
    i, j, k = indices(T)
    p, q, r = get_point(tri, i, j, k)
    cent = DelaunayTriangulation.triangle_centroid(p, q, r)
    push!(centroids, cent)
    u, v, w = DelaunayTriangulation.triangle_edge_midpoints(p, q, r)
    push!(linesegments, cent, u, cent, v, cent, w)
end
linesegments!(ax, linesegments, color=:blue)
scatter!(ax, centroids, color=:red, markersize=8)
```

```@example ex_tri 
fig #hide
```

Let us now establish some notation for referring to these control volumes, using the figure below 
as a reference.

| Symbol | Description | Example |
| :---   | :---        | :---    |
| $\vb x_i$ | A vertex of $\mathcal T(\Omega)$ | The blue point below |
| $\Omega_i$ | The control volume around $\vb x_i$ | The green region below |
| $\partial\Omega_i$ | The boundary of $\Omega_i$ | The blue edges below |
| $V_i$ | The volume of $\Omega_i$ | The volume of the green region below |
| $\mathcal E_i$ | The set of edges of $\partial\Omega_i$ | The blue edges below |
| $\sigma$ | An edge $\sigma \in \mathcal E_i$ | The magenta edge below. Note that $\bigcup_{\sigma \in \mathcal E_i} \sigma = \partial\Omega_i$ |
| $\vb x_{\sigma}$ | The midpoint of $\sigma \in \mathcal E_i$ | The blue point below on $\sigma$ |
| $\hat{\vb n}_{\sigma}$ | The outward unit normal vector to $\sigma \in \mathcal E_i$ | The black arrow below |
| $\mathcal T_i$ | The set of triangles that have $\vb x_i$ as a vertex | The black triangles surrounding $\vb x_i$ below |
| $L_\sigma$ | The length of $\sigma \in \mathcal E_i$ | The length of the magenta edge below

```@setup cv_notation
using DelaunayTriangulation, CairoMakie, LinearAlgebra
a, b, c, d, e, f = (0.0, 0.0),
(2.0, 3.0),
(4.0, -1.0),
(1.0, -4.0),
(-4.0, -2.0),
(-4.0, 2.0)
points = [a, b, c, d, e, f]
tri = triangulate(points)

centroids = NTuple{2,Float64}[]
linesegments = NTuple{2,Float64}[]
other_linesegments = NTuple{2,Float64}[]
non_cv_midpoints = collect.([
    (f .+ b) ./ 2,
    (b .+ c) ./ 2,
    (c .+ d) ./ 2,
    (d .+ e) ./ 2,
    (e .+ f) ./ 2
])
for T in each_solid_triangle(tri)
    i, j, k = indices(T)
    p, q, r = get_point(tri, i, j, k)
    cent = DelaunayTriangulation.triangle_centroid(p, q, r)
    push!(centroids, cent)
    u, v, w = DelaunayTriangulation.triangle_edge_midpoints(p, q, r)
    any(≈(collect(u)), non_cv_midpoints) ? push!(other_linesegments, cent, u) : push!(linesegments, cent, u)
    any(≈(collect(v)), non_cv_midpoints) ? push!(other_linesegments, cent, v) : push!(linesegments, cent, v)
    any(≈(collect(w)), non_cv_midpoints) ? push!(other_linesegments, cent, w) : push!(linesegments, cent, w)
end

control_volume_verts = vcat(linesegments, centroids)
cv_idx = collect(eachindex(control_volume_verts))
DelaunayTriangulation.sort_convex_polygon!(cv_idx, control_volume_verts)
fig, ax, sc = poly(control_volume_verts[cv_idx],
    color=:lightgreen,
    axis=(width=400, height=400))
triplot!(tri,
    show_convex_hull=false,
    show_constrained_edges=false,
    show_all_points=true,
    point_color=:black,
    markersize=11)
scatter!(ax, [a, b, c], color=:cyan, markersize=11, strokecolor=:black, strokewidth=1)
linesegments!(ax, linesegments, color=:blue, linewidth=3)
linesegments!(ax, other_linesegments, color=(:grey, 0.5))

cidx = findfirst(≈(collect((f .+ a .+ b) ./ 3)), collect.(centroids))
lidx = findfirst(≈(collect((f .+ a) ./ 2)), collect.(linesegments))
u, v = centroids[cidx], linesegments[lidx]
rot90_uv = [0 -1; 1 0] * collect(u .- v)
rot90_uv = 0.8rot90_uv ./ norm(rot90_uv)
m = (u .+ v) ./ 2
p = scatter!(ax, [m], color=:blue, markersize=11, strokecolor=:black, strokewidth=1)
arrows!(ax, [m[1]], [m[2]], [rot90_uv[1]], [rot90_uv[2]], color=:black)
lines!(ax, [u, v], color=:magenta, linewidth=3)
translate!(p, 0, 0, 1)

scatter!(ax, centroids, color=:red, markersize=11, strokecolor=:black, strokewidth=1)
text!(ax, -0.8, -1.7, text=L"\Omega_i", fontsize=24)
text!(ax, -3.2, -1.0, text=L"\partial\Omega_i", fontsize=24)
text!(ax, -2.0, 1.1, text=L"\sigma", fontsize=24)
text!(ax, -2.4, 1.6, text=L"\hat{\mathbf{n}}_{\sigma}", fontsize=24)
text!(ax, -1.4, 0.8, text=L"\mathbf{x}_{\sigma}", fontsize=24)
text!(ax, 0.2, 0.0, text=L"\mathbf{x}_{i}", fontsize=24)
text!(ax, 0.2, -0.5, text=L"v_{k1}", fontsize=24)
text!(ax, 3.1, -1.3, text=L"v_{k2}", fontsize=24)
text!(ax, 2.1, 2.9, text=L"v_{k3}", fontsize=24)
text!(ax, 1.7, 1.5, text=L"T_k", fontsize=24)

hidedecorations!(ax)
resize_to_layout!(fig)
```

```@example cv_notation
fig #hide
```

## Discretising the PDE 

Now that we have our concept of control volumes, we can discretise the PDE \eqref{eq:pde}. We do this 
by considering each PDE inside each $\Omega_i$ and integrating. For a given control volume $\Omega_i$, we can integrate the PDE to give
```math 
\begin{equation}\label{eq:integratedi}
\dv{t}\iint_{\Omega_i} u\dd{V} + \iint_{\Omega_i} \div\vb q \dd{V} = \iint_{\Omega_i} S \dd{V}.
\end{equation}
```
Using the divergence theorem, the second integral in \eqref{eq:integratedi} becomes
```math
\begin{equation}\label{eq:applieddthm}
\iint_{\Omega_i} \div\vb q = \oint_{\partial\Omega_i} \vb q \vdot \vu n_\sigma \dd{s} = \sum_{\sigma \in \mathcal E_i} \int_\sigma \vb q \vdot \vu n_\sigma \dd{s},
\end{equation}
```
where the last equality in \eqref{eq:applieddthm} follows from integrating over each individual line segment that defines $\partial\Omega_i$, which is simply $\mathcal E_i$. We then define the _control volume averages_,
```math
\begin{equation}\label{eq:averages}
\bar u_i = \frac{1}{V_i}\iint_{\Omega_i} u\dd{V},\quad \bar S_i = \frac{1}{V_i}\iint_{\Omega_i} S\dd{V},
\end{equation}
```
so that our integral formulation \eqref{eq:integratedi} becomes
```math
\begin{equation}\label{eq:intform}
\dv{\bar u_i}{t} + \frac{1}{V_i}\sum_{\sigma\in\mathcal E_i}\int_\sigma \vb q \vdot \vu n_\sigma \dd{s} = \bar S_i.
\end{equation}
```
Note that \eqref{eq:intform} is still an exact expression.

To proceed, we need to approximate the integrals $\int_\sigma \vb q \vdot \vu n_\sigma\dd{s}$. To accomplish this, we use a midpoint rule, writing
```math
\begin{equation}\label{eq:midpt_rule}
\int_\sigma \vb q \vdot \vu n_\sigma \dd{s} \approx \left[\vb q(\vb x_\sigma, t)\vdot \vu n_\sigma\right]L_\sigma.
\end{equation}
```
Then, replacing the control volume averages with their value at $\vb x_i$, we obtain the following approximation:
```math
\begin{equation}\label{eq:nextapprox}
\dv{u_i}{t} + \frac{1}{V_i}\sum_{\sigma\in\mathcal E_i} \left[\vb q(\vb x_\sigma, t) \vdot \vu n_\sigma\right]L_\sigma = S_i,
\end{equation}
```
where $u_i = u(\vb x_i, t)$ and $S_i = S(\vb x_i, t)$. 

The final step in this part of the approximation is the evaluation of $\vb q(\vb x_\sigma, t)$. To deal with this function, consider some $T_k \in \mathcal T_i$ so that $\vb x_\sigma$ is inside $T_k$. We use a linear shape function inside $T_k$ to approximate $u$, writing
```math
\begin{equation}\label{eq:shape}
u(\vb x, t) = \alpha_kx + \beta_ky + \gamma_k, \quad \vb x \in T_k,
\end{equation}
```
where we have suppressed the dependence of the coefficients $(\alpha_k, \beta_k,\gamma_k)$ on $t$. The vertices of $T_k$ are given by $v_{k1}, v_{k2}, v_{k3}$ with corresponding coordinates $\vb x_{k1}, \vb x_{k2}, \vb x_{k3}$, respectively. We can then solve for the coefficients in \eqref{eq:shape} by requiring that $u$ is equal to the values at the vertices of $T_k$, giving the system of equations
```math
\begin{equation}\label{eq:near_cramer}
\begin{aligned}
u(\vb x_{k1}, t) &= \alpha_kx_{k1} + \beta_ky_{k1} + \gamma_k, \\
u(\vb x_{k2}, t) &= \alpha_kx_{k2} + \beta_ky_{k2} + \gamma_k, \\
u(\vb x_{k3}, t) &= \alpha_kx_{k3} + \beta_ky_{k3} + \gamma_k,
\end{aligned}
\end{equation}
```
where $\vb x_{ki} = (x_{ki}, y_{ki})^{\mkern-1.5mu\mathsf{T}}$. Note that the values on the left-hand side of \eqref{eq:near_cramer} are all known from either the initial condition or the previous time-step. Using Cramer's rule, we define
```math
\begin{equation}\label{eq:shape_coeffs}
\vb S_k = \frac{1}{\Delta_k}\begin{bmatrix} y_{k2}-y_{k3} & y_{k3}-y_{k1} & y_{k1}-y_{k2} \\ x_{k3} - x_{k2} & x_{k1}-x_{k3}&x_{k2}-x_{k1} \\ x_{k2}y_{k3}-x_{k3}y_{k2} & x_{k3}y_{k1}-x_{k1}y_{k3}&x_{k1}y_{k2}-x_{k2}y_{k1} \end{bmatrix},
\end{equation}
```
where
```math
\begin{equation}\label{eq:deltak}
\Delta_k = x_{k1}y_{k2}-x_{k2}y_{k1}-x_{k1}y_{k3}+x_{k3}y_{k1}+x_{k2}y_{k3}-x_{k3}y_{k2},
\end{equation}
```
and thus we obtain
```math
\begin{equation}\label{eq:shapecoeffvals}
\begin{aligned}
\alpha_k &= s_{k,11}u_{k1} + s_{k,12}u_{k2} + s_{k,13}u_{k3}, \\
\beta_k &= s_{k,21}u_{k1} + s_{k,22}u_{k2} + s_{k,23}u_{k3}, \\
\gamma_k &= s_{k,31}u_{k1} + s_{k,32}u_{k2} + s_{k,33}u_{k3},
\end{aligned}
\end{equation}
```
where $u_{ki} = u(\vb x_{ki}, t)$ and $s_{k,ij}$ are the elements of $\vb S_k$. With \eqref{eq:shape} and \eqref{eq:shapecoeffvals}, we can approximate $\vb q(\vb x_\sigma, t)$ and thus obtain the approximation
```math
\begin{equation}\label{eq:interiorapproximation}
\dv{u_i}{t} + \frac{1}{V_i}\sum_{\sigma\in\mathcal E_i} \left[\vb q\left(\vb x_\sigma, t, \alpha_{k(\sigma)}x_\sigma + \beta_{k(\sigma)}y_\sigma + \gamma_k\right)\vdot \vu n_\sigma\right]L_\sigma = S_i,
\end{equation}
```
where $k(\sigma)$ is the index of the triangle that contains $\vb x_\sigma$. This linear shape function also allows us to compute gradients like $\grad u(\vb x_\sigma, t)$, since $\grad u(\vb x_\sigma, t) = (\alpha_{k(\sigma)}, \beta_{k(\sigma)})^{\mkern-1.5mu\mathsf{T}}$.

# Boundary Conditions

Let us now discuss how boundary conditions (BCs) are handled. We assume that BCs take on any of the following forms:
```math
\begin{align}
\vb q(\vb x, t) \vdot \vu n_\sigma &= a(\vb x, t, u) & \vb x \in \mathcal B \subseteq \partial\Omega, \label{eq:neumann} \\
\dv{u(\vb x, t)}{t} &= a(\vb x, t, u) & \vb x \in \mathcal B \subseteq \partial\Omega, \label{eq:dudtdirichlet} \\
u(\vb x, t) &= a(\vb x, t, u) & \vb x \in \mathcal B \subseteq \partial\Omega, \label{eq:dirichlet}
\end{align}
```
where the functions $a$ are user-provided functions. The conditions \eqref{eq:neumann}--\eqref{eq:dirichlet} are called _Neumann_, _time-dependent Dirichlet_, and _Dirichlet_, respectively. We discuss how we handle incompatible BCs below, and then how each of these three types are implemented. 

## Incompatible boundary conditions

When a user specifies BCs, they are allowed to specify different BCs on different parts of the boundary. One issue with allowing this is that the BCs \eqref{eq:neumann}--\eqref{eq:dirichlet} are not compatible with each other. For example, suppose that we have two consecutive edges $\vb e_1 = \overrightarrow{\vb p\vb q}$ and $\vb e_2 = \overrightarrow{\vb q\vb r}$. If there is a Dirichlet BC on $\vb e_1$ and a Neumann BC on $\vb e_2$, then in general it is not possible to satisfy both of these conditions at $\vb q$. To deal with this, we would instead impose a Dirichlet BC on $\vb e_2$.

The rules we apply in general are as follows: Given two oriented edges $\vb e_1 = \overrightarrow{\vb p\vb q}$ and $\vb e_2 = \overrightarrow{\vb q\vb r}$, with corresponding BCs $\mathcal B_1$ and $\mathcal B_2$:
1. If $\mathcal B_1$ or $\mathcal B_2$ is a Dirichlet BC \eqref{eq:dirichlet}, then $\mathcal B_2$ is set to be a Dirichlet BC.
2. If $\mathcal B_1$ or $\mathcal B_2$ is  a time-dependent Dirichlet BC \eqref{eq:dudtdirichlet}, then $\mathcal B_2$ is set to be a time-dependent Dirichlet BC.
3. If $\mathcal B_1$ and $\mathcal B_2$ are Neumann BCs \eqref{eq:neumann}, then $\mathcal B_2$ is set to be a Neumann BC.

Note that the functions that are used for the BCs in the rules above, if there is a replacement, are those defining the BC $\mathcal B_1$.

## Dirichlet boundary conditions

When we have a Dirichlet BC of the form \eqref{eq:dirichlet}, the implementation is simple: Rather than using \eqref{eq:interiorapproximation}, we instead leave $\mathrm du_i/\mathrm dt = 0$ and update the value of $u_i$ with $a(\vb x_i, t, u_i)$ at the end of the iteration using a callback; note that the expression $u_i = a(\vb x_i, t, u_i)$ is __not__ an implicit equation for $u_i$, rather it is simply a reassignment of $u_i$ to $a(\vb x_i, t, u_i)$, i.e. $u_i \leftarrow a(\vb x_i, t, u_i)$.

## Time-dependent Dirichlet boundary conditions 

For a time-dependent Dirichlet BC of the form \eqref{eq:dudtdirichlet}, the implementation is again simple: Rather than using \eqref{eq:interiorapproximation}, simply compute $\mathrm du_i/\mathrm dt = a(\vb x_i, t, u_i)$ instead. 

## Neumann boundary conditions

Neumann boundary conditions \eqref{eq:neumann} are the most complex out of the three. Let us return to our integral formulation \eqref{eq:intform}. Let $\mathcal E_i^n$ be the set of edges in $\mathcal E_i$ that have a Neumann BC associated with them, and $\mathcal E_i^c = \mathcal E_i \setminus \mathcal E_i^n$. Then, also using \eqref{eq:interiorapproximation}, in the case of \eqref{eq:neumann} we can write
```math
\begin{equation}\label{eq:neumanndecomp}
\dv{u_i}{t} + \frac{1}{V_i}\sum_{\sigma\in \mathcal E_i^c} \left[\vb q(\vb x_\sigma, t, \alpha_{k(\sigma)}x_\sigma + \beta_{k(\sigma)}y_\sigma + \gamma_k) \vdot \vu n_\sigma\right]L_\sigma + \frac{1}{V_i}\sum_{\sigma\in\mathcal E_i^n} \int_{\sigma} a_{\sigma}(\vb x, t, u)\dd{s} = S_i,
\end{equation}
```
where $a_\sigma$ is the BC function associated with $\sigma$. This integral is then approximated using a midpoint rule as done previously, giving
```math
\begin{equation}\label{eq:neumanndecompapprox}
\dv{u_i}{t} + \frac{1}{V_i}\sum_{\sigma\in \mathcal E_i^c} \left[\vb q(\vb x_\sigma, t, u(\vb x_\sigma, t)) \vdot \vu n_\sigma\right]L_\sigma + \frac{1}{V_i}\sum_{\sigma\in\mathcal E_i^n} \left[a_{\sigma}(\vb x_\sigma, t, u(\vb x_\sigma, t))\right]L_\sigma = S_i,
\end{equation}
```
where $u(\vb x_\sigma, t) = \alpha_{k(\sigma)}x_\sigma + \beta_{k(\sigma)}y_\sigma + \gamma_k$.

# Internal Conditions

We also allow for specifying internal conditions, meaning conditions of the form \eqref{eq:neumann}--\eqref{eq:dirichlet} that are applied away from the boundary. We do not currently allow for internal Neumann conditions.[^1] These conditions are handled in the same way as BCs, except that the user is to provide them per-vertex rather than per-edge.

[^1]: This is a technical limitation due to how the control volumes are defined. For vertices away from the boundary, the control volume edges do not lie along any of the triangle's edges, which is where we would like to impose Neumann conditions.


# Putting Everything Together 

We have now specified how we discretise the PDE itself, and how we handle both boundary and internal conditions. The remaining task is to actually discuss how we compute each $\mathrm du_i/\mathrm dt$. As written, \eqref{eq:interiorapproximation} indicates that we loop over each vertex and, within each vertex, loop over each edge of its control volume. This would not be the most efficient way to do this when actually implementing it in code. Instead, we loop over each triangle in $\mathcal T(\Omega)$, picking up the contribution from each control volume edge inside the triangle to the system of equations for $\mathrm du_i/\mathrm dt$.

To understand how this can be done, consider the figure below.

```@setup ex_focus
using DelaunayTriangulation, CairoMakie
a, b, c, d, e, f, g, h, i, j, k, ℓ = (0.0, 0.0),
(2.0, 3.0),
(4.0, -1.0),
(1.0, -4.0),
(-4.0, -2.0),
(-4.0, 2.0),
(-1.0, 5.0),
(6.0, 3.0),
(7.0, -2.0),
(4.0, -5.0),
(-3.0, -4.0),
(-6.0, 0.0)
points = [a, b, c, d, e, f, g, h, i, j, k, ℓ]
tri = triangulate(points)

fig = Figure()
ax = Axis(fig[1, 1], width=400, height=400)
for i in 1:3
    p = get_point(tri, i)
    cv = NTuple{2,Float64}[]
    S = DelaunayTriangulation.get_surrounding_polygon(tri, i)
    push!(S, S[begin])
    j = S[begin]
    q = get_point(tri, j)
    push!(cv, (p .+ q) ./ 2)
    for k in S[2:end]
        r = get_point(tri, k)
        push!(cv, (p .+ q .+ r) ./ 3)
        push!(cv, (p .+ r) ./ 2)
        q = r
    end
    L = lines!(ax, cv, color=:blue, linewidth=3)
    translate!(L, 0, 0, 1)
end
poly!(ax, [a, c, b],
    color=:lightblue)
triplot!(tri,
    show_convex_hull=false,
    show_constrained_edges=false)
scatter!(ax, [a, c, b], color=[:blue, :red, :darkgreen], strokecolor=:black, strokewidth=1)
text!(ax, 2.1, -0.3, text=L"\sigma_{br}", fontsize=24)
text!(ax, 2.1, 1.0, text=L"\sigma_{rg}", fontsize=24)
text!(ax, 1.0, 0.6, text=L"\sigma_{gb}", fontsize=24)
br = lines!(ax, [(a .+ c) ./ 2, (a .+ b .+ c) ./ 3], color=:purple, linewidth=3)
translate!(br, 0, 0, 1)
bg = lines!(ax, [(a .+ b) ./ 2, (a .+ b .+ c) ./ 3], color=:teal, linewidth=3)
translate!(bg, 0, 0, 1)
rg = lines!(ax, [(b .+ c) ./ 2, (a .+ b .+ c) ./ 3], color=:yellow, linewidth=3)
translate!(rg, 0, 0, 1)
xlims!(ax, -3, 5.9)
ylims!(ax, -3.5, 4.5)
fig
hidedecorations!(ax)
resize_to_layout!(fig)
```

```@example ex_focus
fig #hide
```

We denote the triangle in blue by $T$, and refer to the blue, red, and green vertices by $v_b$, $v_r$, and $v_g$, respectively. The relevant edges that contribute to $\mathrm du_b/\mathrm dt$, $\mathrm du_r/\mathrm dt$, and $\mathrm du_g/\mathrm dt$ are $\sigma_{br}$, $\sigma_{rg}$, and $\sigma_{gb}$, as annotated above. In particular, $\sigma_{br}$ contributes to both $\mathrm du_b/\mathrm dt$ and $\mathrm du_r/\mathrm dt$, $\sigma_{rg}$ contributes to both $\mathrm du_r/\mathrm dt$ and $\mathrm du_g/\mathrm dt$, and $\sigma_{gb}$ contributes to both $\mathrm du_g/\mathrm dt$ and $\mathrm du_b/\mathrm dt$. 

Let us focus on $u_b$ and $u_r$. The procedure for going from the edge $e_{br}$ to getting the contribution to $\mathrm du_b/\mathrm dt$ and $\mathrm du_r/\mathrm dt$ is as follows:

1. Compute the midpoint of $e_{br}$, denoted $\vb x_{br}$.
2. Compute the centroid of $T$, given by $\vb x_T$; this should only be done once for $T$, so for the next two edges this can be stored.
3. Compute the unit normal vector $\vu n_{br}$ to $\overrightarrow{\vb x_{br}\vb x_T}$, which is the unit normal vector to $\sigma_{br}$. This will be pointing away from $\vb x_b$, but towards $\vb x_r$, so we will need to be careful of the orientation of $\vu n_{br}$ relative to the respective control volumes of $v_b$ and $v_r$.
4. Compute $L_{br} = \|\vb x_T - \vb x_{br}\|$.
5. Compute $\alpha$, $\beta$, and $\gamma$ from \eqref{eq:shapecoeffvals} using the vertices of $T$.
6. Let $Q = [\vb q(\vb x_{br}, t, \alpha x_{br} + \beta y_{br} + \gamma) \vdot \vu n_{br}]L_{br}$.
7. Perform the reassignment $\mathrm du_b/\mathrm dt \leftarrow \mathrm du_b/\mathrm dt - Q$. Note that the minus sign is because we have brought the sum in \eqref{eq:interiorapproximation} to the left-hand side of the equation.
8. Perform the reassignment $\mathrm du_r/\mathrm dt \leftarrow \mathrm du_r/\mathrm dt + Q$. Note that this second term is positive because $\vu n_{br}$ is pointing towards $\vb x_r$.

With this procedure, we can get the contribution from each edge without having to repeat many computations. The procedure for the other edges is similar. Care does need to be taken if there is a boundary or internal condition on $u_b$ or $u_r$ (or the other vertices). If there is a Dirichlet condition on $u_b$ (either \eqref{eq:dudtdirichlet} or \eqref{eq:dirichlet}), skip step 7. Similarly, skip step 8 if there is a Dirichlet condition on $u_r$. If there is a Neumann condition on $e_{br}$, then the definition of $Q$ should be modified to be $Q = [a_{br}(\vb x_{br}, t, \alpha x_{br} + \beta y_{br} + \gamma)]L_{br}$, where $a_{br}$ is the Neumann condition function associated with $e_{br}$; it does not matter if, for example, $u_b$ is a part of a Neumann condition for some other edge, since Neumann conditions are specified per-edge rather than per-vertex.

Once we have applied these steps onto each triangle, and we have taken care for respecting the boundary and internal conditions, the last step is to loop over each vertex and then compute the source term $S_i$, adding it to each equation. This completes the discretisation of the PDE.