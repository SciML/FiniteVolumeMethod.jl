```@meta
EditURL = "https://github.com/DanielVandH/FiniteVolumeMethod.jl/tree/main/docs/src/literate_tutorials/diffusion_equation_on_an_annulus.jl"
```


# Diffusion Equation on an Annulus
In this tutorial, we consider a
diffusion equation on an annulus:
```math
\begin{equation}
\begin{aligned}
\pdv{u(\vb x, t)}{t} &= \grad^2 u(\vb x, t) & \vb x \in \Omega, \\
\grad u(\vb x, t) \vdot \vu n(\vb x) &= 0 & \vb x \in \mathcal D(0, 1), \\
u(\vb x, t) &= c(t) & \vb x \in \mathcal D(0,0.2), \\
u(\vb x, t) &= u_0(\vb x),
\end{aligned}
\end{equation}
```
demonstrating how we can solve PDEs over multiply-connected domains.
Here, $\mathcal D(0, r)$ is a circle of radius $r$ centred at the origin,
$\Omega$ is the annulus between $\mathcal D(0,0.2)$ and
$\mathcal D(0, 1)$, $c(t) = 50[1-\mathrm{e}^{-t/2}]$, and
```math
u_0(x) = 10\mathrm{e}^{-25\left[\left(x+\frac12\right)^2+\left(y+\frac12\right)^2\right]} - 10\mathrm{e}^{-45\left[\left(x-\frac12\right)^2+\left(y-\frac12\right)^2\right]} - 5\mathrm{e}^{-50\left[\left(x+\frac{3}{10}\right)^2+\left(y+\frac12\right)^2\right]}.
```
The complicated task for this problem is the definition
of the mesh of the annulus. We need to follow the boundary
specification from DelaunayTriangulation.jl, discussed
[here](https://danielvandh.github.io/DelaunayTriangulation.jl/dev/boundary_handling/).
In particular, the outer boundary must be counter-clockwise,
the inner boundary be clockwise, and we need to provide
the nodes as a `Vector{Vector{Vector{Int}}}`.
We define this mesh below.

````julia
using DelaunayTriangulation, FiniteVolumeMethod, CairoMakie
R₁ = 0.2
R₂ = 1.0
θ = collect(LinRange(0, 2π, 100))
θ[end] = 0.0 # get the endpoints to match
x = [
    [R₂ .* cos.(θ)], # outer first
    [reverse(R₁ .* cos.(θ))] # then inner - reverse to get clockwise orientation
]
y = [
    [R₂ .* sin.(θ)], #
    [reverse(R₁ .* sin.(θ))]
]
boundary_nodes, points = convert_boundary_points_to_indices(x, y)
tri = triangulate(points; boundary_nodes)
A = get_total_area(tri)
refine!(tri; max_area=1e-4A)
triplot(tri)
````
![](diffusion_equation_on_an_annulus-4.png)

````julia
mesh = FVMGeometry(tri)
````

````
FVMGeometry with 8275 control volumes, 16109 triangles, and 24384 edges
````

Now let us define the boundary conditions. Remember,
the order of the boundary conditions follows the order
of the boundaries in the mesh. The outer boundary
came first, and then came the inner boundary. We can verify
that this is the order of the boundary indices as
follows:

````julia
fig = Figure()
ax = Axis(fig[1, 1])
outer = [get_point(tri, i) for i in get_neighbours(tri, -1)]
inner = [get_point(tri, i) for i in get_neighbours(tri, -2)]
triplot!(ax, tri)
scatter!(ax, outer, color=:red)
scatter!(ax, inner, color=:blue)
fig
````
![](diffusion_equation_on_an_annulus-7.png)

So, the boundary conditions are:

````julia
outer_bc = (x, y, t, u, p) -> zero(u)
inner_bc = (x, y, t, u, p) -> oftype(u, 50(1 - exp(-t / 2)))
types = (Neumann, Dirichlet)
BCs = BoundaryConditions(mesh, (outer_bc, inner_bc), types)
````

````
BoundaryConditions with 2 boundary conditions with types (Neumann, Dirichlet)
````

Finally, let's define the problem and solve it.

````julia
initial_condition_f = (x, y) -> begin
    10 * exp(-25 * ((x + 0.5) * (x + 0.5) + (y + 0.5) * (y + 0.5))) - 5 * exp(-50 * ((x + 0.3) * (x + 0.3) + (y + 0.5) * (y + 0.5))) - 10 * exp(-45 * ((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)))
end
diffusion_function = (x, y, t, u, p) -> one(u)
initial_condition = [initial_condition_f(x, y) for (x, y) in each_point(tri)]
final_time = 2.0
prob = FVMProblem(mesh, BCs;
    diffusion_function,
    final_time,
    initial_condition)
````

````
FVMProblem with 8275 nodes and time span (0.0, 2.0)
````

````julia
using OrdinaryDiffEq, LinearSolve
sol = solve(prob, TRBDF2(linsolve=KLUFactorization()), saveat=0.2)
````

````
retcode: Success
Interpolation: 1st order linear
t: 11-element Vector{Float64}:
 0.0
 0.2
 0.4
 0.6
 0.8
 1.0
 1.2
 1.4
 1.6
 1.8
 2.0
u: 11-element Vector{Vector{Float64}}:
 [-1.6918979226151304e-9, -2.6824824758316947e-8, -3.508918015123032e-7, -3.7478900338780008e-6, -3.2376946343892575e-5, -0.0002242588591250139, -0.0012357868125636464, -0.005380630921148495, -0.01840119235904892, -0.04918487388571497  …  -8.453425926424107e-13, 2.8396329156286174e-11, -0.987027875279866, 1.1090808318535561e-14, 0.007525438950030774, 6.391588915642178e-5, 5.288433070362081e-5, 4.1020493095240526e-5, -0.0004993269323277568, 1.241301762751446]
 [0.044756806800383464, 0.01697807113633325, -0.009934249089260739, -0.035167807974441155, -0.05909026293562593, -0.08118131359768818, -0.10128253287478403, -0.11909644940569893, -0.1340531809976604, -0.1462196255124722  …  0.484190844762769, 0.5532803099645588, 0.4684565815930154, 0.4187513669494636, 1.8088917321252063, 1.2806475811239213, 0.9517124945963161, 0.9298132438059296, 3.022655530508937, 1.4178171713937289]
 [1.8560195570424072, 1.844631378542109, 1.8339134321531902, 1.824841375549887, 1.8162192570014926, 1.808617023404368, 1.8017917968761172, 1.7958306635091614, 1.7912776583905223, 1.7876067418613069  …  2.4034195811269825, 2.123458670272011, 2.751390565847567, 2.178291558722974, 4.216107162022562, 3.3934242474696914, 2.615137746140509, 2.5672752042264797, 6.632334068301745, 3.192089588889734]
 [4.384459050958909, 4.379036076855718, 4.373827419842427, 4.3700104051149475, 4.366055560811536, 4.362666689821256, 4.359505593334478, 4.356659143541885, 4.354774551745091, 4.353172062069764  …  4.998749781908298, 4.508929684598627, 5.527534719350656, 4.681907803177377, 7.102509001724943, 6.098203846737156, 5.074001463699793, 5.011333528961685, 10.28645968999563, 5.748760933074804]
 [7.289250720305069, 7.286603926176796, 7.283945386228958, 7.282551484708967, 7.280732935296608, 7.279259737854163, 7.277749802661687, 7.276291949308102, 7.275585615823201, 7.274878697686495  …  7.936084046920031, 7.346744061040773, 8.549282372098798, 7.575530639228972, 10.167136423577562, 9.086260277509481, 7.949479000127506, 7.8798028034681815, 13.52237412603819, 8.671783106247956]
 [10.34979855078777, 10.34842876878896, 10.346946217022017, 10.346652982011351, 10.345812535776243, 10.345213207490707, 10.34445875862652, 10.343637524803464, 10.3434621352153, 10.343158996391354  …  11.001350928760237, 10.376780197524889, 11.641735862783912, 10.626929967344758, 13.247267054361103, 12.154306391125157, 10.987487101967028, 10.915797483301303, 16.603022787182297, 11.719034675091235]
 [13.419032229786552, 13.418248614435283, 13.417311411811106, 13.417501046413571, 13.417104210110661, 13.416897276092282, 13.416487882593142, 13.415961903126915, 13.416019693963312, 13.415897827637506  …  14.05549300910112, 13.432187293618043, 14.688766676806283, 13.685125764113343, 16.22665450012967, 15.173998401637757, 14.030354492344383, 13.959754647440358, 19.247266559020932, 14.744458034987929]
 [16.402808386145264, 16.402299124744406, 16.401624087959547, 16.40201195515885, 16.40181620728577, 16.401782333079368, 16.40153266472225, 16.401149172174552, 16.40130371832505, 16.401262561794372  …  17.011593935218123, 16.409673739663027, 17.620558060073513, 16.655356115055124, 19.079827341462828, 18.07799614506547, 16.982536207788538, 16.91476781933308, 21.926623622603703, 17.66524656994868]
 [19.245440451860773, 19.24506502780147, 19.244524750455852, 19.24498173335047, 19.244877584211643, 19.244917635490616, 19.244744142471863, 19.244434453684597, 19.244623974158994, 19.24461819846793  …  19.820915170511924, 19.249414165476868, 20.399635907557563, 19.48316532785163, 21.792730088982616, 20.83211311080772, 19.791213137966565, 19.726993148981634, 24.70747930614682, 20.438278176816194]
 [21.917127170018293, 21.916822716156265, 21.916360514782337, 21.916827580969215, 21.916765754647194, 21.916834902510207, 21.916700141262016, 21.916433748387934, 21.916630059881474, 21.916639220595982  …  22.45563287728568, 21.91973679283083, 22.999584933963234, 22.1390462009306, 24.319804178687953, 23.406125261547807, 22.42683471906728, 22.36663329136451, 27.25284597805504, 23.034265764524818]
 [24.3993717709466, 24.39911198138617, 24.398705557041268, 24.399152251863452, 24.399110776065974, 24.39918807508382, 24.399076028558895, 24.39884108560826, 24.399029867063096, 24.39904459027539  …  24.895558877067963, 24.401292012101774, 25.401896777370617, 24.603367732010152, 26.67474506860102, 25.785290280312385, 24.86852396146949, 24.812894402533054, 29.776107756155884, 25.43365475051902]
````

````julia
fig = Figure(fontsize=38)
for (i, j) in zip(1:3, (1, 6, 11))
    ax = Axis(fig[1, i], width=600, height=600,
        xlabel="x", ylabel="y",
        title="t = $(sol.t[j])",
        titlealign=:left)
    tricontourf!(ax, tri, sol.u[j], levels=-10:2:40, colormap=:matter)
    tightlimits!(ax)
end
resize_to_layout!(fig)
fig
````
![](diffusion_equation_on_an_annulus-13.png)

To finish this example, let us consider how
natural neighbour interpolation can be applied here. The
application is more complicated for this problem since
the mesh has holes. Before we do that, though, let us
show how we could use `pl_interpolate`, which could
be useful if we did not need a higher quality interpolant.
Let us interpolate the solution at $t = 1$, which
is `sol.t[6]`. For this, we need to put the ghost
triangles back into `tri` so that we can safely
apply `jump_and_march`. This is done with `add_ghost_triangles!`.

````julia
add_ghost_triangles!(tri)
````

Now let's interpolate.

````julia
x = LinRange(-R₂, R₂, 400)
y = LinRange(-R₂, R₂, 400)
interp_vals = zeros(length(x), length(y))
u = sol.u[6]
last_triangle = Ref((1, 1, 1))
for (j, _y) in enumerate(y)
    for (i, _x) in enumerate(x)
        T = jump_and_march(tri, (_x, _y), try_points=last_triangle[])
        last_triangle[] = indices(T) # used to accelerate jump_and_march, since the points we're looking for are close to each other
        if DelaunayTriangulation.is_ghost_triangle(T) # don't extrapolate
            interp_vals[i, j] = NaN
        else
            interp_vals[i, j] = pl_interpolate(prob, T, sol.u[6], _x, _y)
        end
    end
end
fig, ax, sc = contourf(x, y, interp_vals, levels=-10:2:40, colormap=:matter)
fig
````
![](diffusion_equation_on_an_annulus-17.png)

Let's now consider applying NaturalNeighbours.jl. We apply it naively first to
highlight some complications.

````julia
using NaturalNeighbours
_x = vec([x for x in x, y in y]) # NaturalNeighbours.jl needs vector data
_y = vec([y for x in x, y in y])
itp = interpolate(tri, u, derivatives=true)
````

````
Natural Neighbour Interpolant
    z: [10.34979855078777, 10.34842876878896, 10.346946217022017, 10.346652982011351, 10.345812535776243, 10.345213207490707, 10.34445875862652, 10.343637524803464, 10.3434621352153, 10.343158996391354  …  11.001350928760237, 10.376780197524889, 11.641735862783912, 10.626929967344758, 13.247267054361103, 12.154306391125157, 10.987487101967028, 10.915797483301303, 16.603022787182297, 11.719034675091235]
    ∇: [(0.019928316962875017, -0.011771165803560442), (-0.004946993788048853, -0.015791083567318072), (-0.02822420967342779, -0.011410875941405212), (0.05041262971926868, -0.010204082611345826), (0.020656106378514488, -0.00831246594037242), (0.020178352779137165, -0.005841110848905656), (0.02263059184157456, 0.003940682306225492), (-0.015427254865983992, -0.009281370028003864), (0.0011990900623802534, -0.008620269614894408), (0.02015387063370135, 0.007769147484581202)  …  (-4.547307168200022, 2.057744418828956), (-0.14124347717445376, 0.22448276452263805), (-6.764629478886202, -3.4125908109468863), (-2.521964823867538, 1.7205775030247321), (12.665719267528003, -0.9836769351501671), (-2.547112391057987, 8.842938028766415), (4.698728938888856, -1.0991274288151758), (4.375058501487993, -1.0217294908779913), (-2.4521298631357076, -7.655464205569285), (7.145102737799175, 2.733916843292211)]
    H: [(15.791910277191064, -0.33602088095412536, 0.43157323471343956), (14.620002041132855, -0.07852069998855293, 1.3505715314629894), (13.767131378456325, 0.6216554303306571, 1.600568117074863), (17.29865794688917, 0.39522144053367747, 3.370055890604093), (15.449336924884578, 0.6860519477147633, 4.011656375976057), (14.01617882710455, 1.2117376686677876, 5.28149046986194), (14.946581602877059, 1.7778392429517798, 5.7077589568727065), (11.153823789593469, 3.0434742790908884, 5.423404227197588), (11.187660375920457, 3.6186061354887324, 6.397319756211962), (11.621681684040475, 4.080571974296338, 7.40025788287922)  …  (17.491817582040213, -1.604681659128542, -10.982574845763178), (3.3365703993181457, 11.65206023390916, -6.44449770736986), (19.454546653358747, -3.9649066107087916, 15.782147298028232), (11.49781358656056, 3.5642132262531128, -10.740362688453766), (39.069814639408456, -27.18277410969509, -5.026862436668616), (-13.105784156753186, 27.092194361202303, -12.35248775488911), (20.017758856761436, -5.590559527118113, -6.1254539665611905), (20.071922246410157, -4.796516334148717, -5.829475788456827), (-270.28306811109485, 1455.8770103204336, 801.6446484449093), (22.722069763770232, -7.095477165945185, 13.366340483188218)]
````

````julia
itp_vals = itp(_x, _y; method=Farin())
````

````
160000-element Vector{Float64}:
 10.399006900237111
 10.399058246953757
 10.394731195745749
 10.394743770894493
 10.394756346043224
  ⋮
 10.33978561240302
 10.339729969962121
 10.339674327521237
 10.339618685080353
 10.339563042639469
````

````julia
fig, ax, sc = contourf(x, y, reshape(itp_vals, length(x), length(y)), colormap=:matter, levels=-10:2:40)
fig
````
![](diffusion_equation_on_an_annulus-21.png)

The issue here is that the interpolant is trying to extrapolate inside the hole and
outside of the annulus. To avoid this, you need to pass `project=false`.

````julia
itp_vals = itp(_x, _y; method=Farin(), project=false)
````

````
160000-element Vector{Float64}:
 Inf
 Inf
 Inf
 Inf
 Inf
  ⋮
 Inf
 Inf
 Inf
 Inf
 Inf
````

````julia
fig, ax, sc = contourf(x, y, reshape(itp_vals, length(x), length(y)), colormap=:matter, levels=-10:2:40)
fig
````
![](diffusion_equation_on_an_annulus-24.png)

## Just the code
An uncommented version of this example is given below.
You can view the source code for this file [here](https://github.com/DanielVandH/FiniteVolumeMethod.jl/tree/main/docs/src/literate_tutorials/diffusion_equation_on_an_annulus.jl).

```julia
using DelaunayTriangulation, FiniteVolumeMethod, CairoMakie
R₁ = 0.2
R₂ = 1.0
θ = collect(LinRange(0, 2π, 100))
θ[end] = 0.0 # get the endpoints to match
x = [
    [R₂ .* cos.(θ)], # outer first
    [reverse(R₁ .* cos.(θ))] # then inner - reverse to get clockwise orientation
]
y = [
    [R₂ .* sin.(θ)], #
    [reverse(R₁ .* sin.(θ))]
]
boundary_nodes, points = convert_boundary_points_to_indices(x, y)
tri = triangulate(points; boundary_nodes)
A = get_total_area(tri)
refine!(tri; max_area=1e-4A)
triplot(tri)

mesh = FVMGeometry(tri)

fig = Figure()
ax = Axis(fig[1, 1])
outer = [get_point(tri, i) for i in get_neighbours(tri, -1)]
inner = [get_point(tri, i) for i in get_neighbours(tri, -2)]
triplot!(ax, tri)
scatter!(ax, outer, color=:red)
scatter!(ax, inner, color=:blue)
fig

outer_bc = (x, y, t, u, p) -> zero(u)
inner_bc = (x, y, t, u, p) -> oftype(u, 50(1 - exp(-t / 2)))
types = (Neumann, Dirichlet)
BCs = BoundaryConditions(mesh, (outer_bc, inner_bc), types)

initial_condition_f = (x, y) -> begin
    10 * exp(-25 * ((x + 0.5) * (x + 0.5) + (y + 0.5) * (y + 0.5))) - 5 * exp(-50 * ((x + 0.3) * (x + 0.3) + (y + 0.5) * (y + 0.5))) - 10 * exp(-45 * ((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)))
end
diffusion_function = (x, y, t, u, p) -> one(u)
initial_condition = [initial_condition_f(x, y) for (x, y) in each_point(tri)]
final_time = 2.0
prob = FVMProblem(mesh, BCs;
    diffusion_function,
    final_time,
    initial_condition)

using OrdinaryDiffEq, LinearSolve
sol = solve(prob, TRBDF2(linsolve=KLUFactorization()), saveat=0.2)

fig = Figure(fontsize=38)
for (i, j) in zip(1:3, (1, 6, 11))
    ax = Axis(fig[1, i], width=600, height=600,
        xlabel="x", ylabel="y",
        title="t = $(sol.t[j])",
        titlealign=:left)
    tricontourf!(ax, tri, sol.u[j], levels=-10:2:40, colormap=:matter)
    tightlimits!(ax)
end
resize_to_layout!(fig)
fig

add_ghost_triangles!(tri)

x = LinRange(-R₂, R₂, 400)
y = LinRange(-R₂, R₂, 400)
interp_vals = zeros(length(x), length(y))
u = sol.u[6]
last_triangle = Ref((1, 1, 1))
for (j, _y) in enumerate(y)
    for (i, _x) in enumerate(x)
        T = jump_and_march(tri, (_x, _y), try_points=last_triangle[])
        last_triangle[] = indices(T) # used to accelerate jump_and_march, since the points we're looking for are close to each other
        if DelaunayTriangulation.is_ghost_triangle(T) # don't extrapolate
            interp_vals[i, j] = NaN
        else
            interp_vals[i, j] = pl_interpolate(prob, T, sol.u[6], _x, _y)
        end
    end
end
fig, ax, sc = contourf(x, y, interp_vals, levels=-10:2:40, colormap=:matter)
fig

using NaturalNeighbours
_x = vec([x for x in x, y in y]) # NaturalNeighbours.jl needs vector data
_y = vec([y for x in x, y in y])
itp = interpolate(tri, u, derivatives=true)

itp_vals = itp(_x, _y; method=Farin())

fig, ax, sc = contourf(x, y, reshape(itp_vals, length(x), length(y)), colormap=:matter, levels=-10:2:40)
fig

itp_vals = itp(_x, _y; method=Farin(), project=false)

fig, ax, sc = contourf(x, y, reshape(itp_vals, length(x), length(y)), colormap=:matter, levels=-10:2:40)
fig
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

