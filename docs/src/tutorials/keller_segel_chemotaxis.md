# Keller-Segel Chemotaxis 
In this tutorial, we consider the following Keller-Segel model of chemotaxis:
```math 
\begin{equation*}
\begin{aligned}
\pdv{u}{t} &= \grad^2u - \div \left(\frac{cu}{1+u^2}\grad v\right) + u(1-u), \\
\pdv{v}{t} &= D\grad^2 v + u - av,
\end{aligned}
\end{equation*}
```
inside the square $[0, 100]^2$ with homogeneous Neumann boundary conditions. We 
start by defining the problem, remembering that we need one problem for each variable $u$
and $v$.

```@example
using FiniteVolumeMethod, DelaunayTriangulation
tri = triangulate_rectangle(0, 100, 0, 100, 250, 250, single_boundary=true)
mesh = FVMGeometry(tri)
bc_u = (x, y, t, (u, v), p) -> zero(u)
bc_v = (x, y, t, (u, v), p) -> zero(v)
BCs_u = BoundaryConditions(mesh, bc_u, Neumann)
BCs_v = BoundaryConditions(mesh, bc_v, Neumann)
q_u = (x, y, t, (αu, αv), (βu, βv), (γu, γv), p) -> begin
    u = αu * x + βu * y + γu
    ∇u = (αu, βu)
    ∇v = (αv, βv)
    χu = p.c * u / (1 + u^2)
    _q = χu .* ∇v .- ∇u
    return _q
end
q_v = (x, y, t, (αu, αv), (βu, βv), (γu, γv), p) -> begin
    ∇v = (αv, βv)
    _q = -p.D .* ∇v
    return _q
end
S_u = (x, y, t, (u, v), p) -> begin
    return u * (1 - u)
end
S_v = (x, y, t, (u, v), p) -> begin
    return u - p.a * v
end
q_u_parameters = (c=4.0,)
q_v_parameters = (D=1.0,)
S_v_parameters = (a=0.1,)
u_initial_condition = 0.01rand(num_points(tri))
v_initial_condition = zeros(num_points(tri))
final_time = 1000.0
u_prob = FVMProblem(mesh, BCs_u;
    flux_function=q_u, flux_parameters=q_u_parameters,
    source_function=S_u,
    initial_condition=u_initial_condition, final_time=final_time)
v_prob = FVMProblem(mesh, BCs_v;
    flux_function=q_v, flux_parameters=q_v_parameters,
    source_function=S_v, source_parameters=S_v_parameters,
    initial_condition=v_initial_condition, final_time=final_time)
prob = FVMSystem(u_prob, v_prob);
```

Now let's solve and animate the problem.
```julia
using OrdinaryDiffEq, LinearSolve, CairoMakie
sol = solve(prob, TRBDF2(linsolve=KLUFactorization()), saveat=1.0) # very slow
fig = Figure(fontsize=44)
x = LinRange(0, 100, 250)
y = LinRange(0, 100, 250)
i = Observable(1)
axu = Axis(fig[1, 1], width=600, height=600,
    title=map(i -> L"u(x,~ y,~ %$(sol.t[i]))", i), xlabel=L"x", ylabel=L"y")
axv = Axis(fig[1, 2], width=600, height=600,
    title=map(i -> L"v(x,~ y,~ %$(sol.t[i]))", i), xlabel=L"x", ylabel=L"y")
u = map(i -> reshape(sol.u[i][1, :], 250, 250), i)
v = map(i -> reshape(sol.u[i][2, :], 250, 250), i)
heatmap!(axu, x, y, u, colorrange=(0.0, 2.5), colormap=:turbo)
heatmap!(axv, x, y, v, colorrange=(0.0, 10.0), colormap=:turbo)
resize_to_layout!(fig)
record(fig, joinpath(@__DIR__, "../figures", "keller_segel_chemotaxis.mp4"), eachindex(sol);
    framerate=60) do _i
    i[] = _i
end;
```

```@raw html
<figure>
    <img src='../figures/keller_segel_chemotaxis.mp4', alt='Animation of the solution'><br>
</figure>
```

Some pretty amazing patterns!