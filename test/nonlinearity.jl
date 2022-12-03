function triangle_to_mat(T)
    _T = collect(T)
    V = [_T[i][j] for i in eachindex(_T), j in 1:3]
    return V
end
using CairoMakie
function evaluate_error(interp, xy, pts, adj, adj2v, DG, r)
    rand() < 0.05 && @show rand(), _i[], xy
    q = xy
    fill!(r, 0.0)
    ## Do the approx first
    k = DelaunayTriangulation.select_initial_point(pts, q; try_points=trypt[])
    T = DelaunayTriangulation.jump_and_march(q, adj, adj2v, DG, pts; k=k)
    if DelaunayTriangulation.is_ghost_triangle(T)
        T = DelaunayTriangulation.rotate_ghost_triangle_to_boundary_form(T)
        u, v, _ = T
        w = DelaunayTriangulation.get_edge(adj, v, u)
        T = (v, u, w)
    end
    trypt[] = T
    if T ∈ keys(interp[1])
        for i in eachindex(interp)
            r[i] = eval_interpolant(interp[i], xy[1], xy[2], T)
        end
        return nothing
    else
        T = DelaunayTriangulation.shift_triangle_1(T)
        if T ∈ keys(interp[1])
            for i in eachindex(interp)
                r[i] = eval_interpolant(interp[i], xy[1], xy[2], T)
            end
            return nothing
        else
            T = DelaunayTriangulation.shift_triangle_1(T)
            for i in eachindex(interp)
                r[i] = eval_interpolant(interp[i], xy[1], xy[2], T)
            end
        end
    end
    return nothing
end#function evaluate_error(interp, interp_exact, xy, pts, adj, adj2v, DG, _pts, _adj, _adj2v, _DG, r)

function integrate_solution(interp, interp_exact, pts, adj, adj2v, DG, _pts, _adj, _adj2v, _DG; xmin=(0.0, 0.0), xmax=(2.0, 2.0))
    val, _ = hcubature(length(interp),
        (xy, r) -> evaluate_error(interp, interp_exact, xy, pts, adj, adj2v, DG, _pts, _adj, _adj2v, _DG, r), xmin, xmax; abstol=1e-6, reltol=1e-6)
    r = zeros(length(interp))
    return val / ((xmax[1] - xmin[1]) * (xmax[2] - xmin[2]))
end

function sum_error(interp, pts, adj, adj2v, DG, _pts, sol)
    r = zeros(length(interp))
    num_pts = size(_pts, 2)
    errs = zeros(length(interp))
    for i in axes(_pts, 2)
        evaluate_error(interp, _pts[:, i], pts, adj, adj2v, DG, r)
        for j in eachindex(interp)
            errs[j] += (r[j] - sol.u[j][i])^2
        end
    end
    return errs ./ num_pts
end

########### Linearly singular diffusion
function evaluate_linearly_singular_problem(r)
    ## Generate the mesh 
    iip_flux = true
    x₁ = LinRange(0, 2, 1000)
    y₁ = LinRange(0, 0, 1000)
    x₂ = LinRange(2, 2, 1000)
    y₂ = LinRange(0, 2, 1000)
    x₃ = LinRange(2, 0, 1000)
    y₃ = LinRange(2, 2, 1000)
    x₄ = LinRange(0, 0, 1000)
    y₄ = LinRange(2, 0, 1000)
    x = reduce(vcat, [x₁, x₂, x₃, x₄])
    y = reduce(vcat, [y₁, y₂, y₃, y₄])
    xy = [[x[i], y[i]] for i in eachindex(x)]
    unique!(xy)
    x = [xy[i][1] for i in eachindex(xy)]
    y = [xy[i][2] for i in eachindex(xy)]
    T, adj, adj2v, DG, pts, BN = generate_mesh(x, y, r; gmsh_path=GMSH_PATH)
    ## Define the problem 
    mesh = FVMGeometry(T, adj, adj2v, DG, pts, BN)
    boundary_functions = Vector{Function}([(x, y, t, p) -> x < 1.0 ? 5.0 : 10.0])
    type = ["Dirichlet"]
    boundary_conditions = BoundaryConditions(mesh, boundary_functions, type)
    f = (x, y) -> 7.5
    D = (x, y, t, u, p) -> 8.0 / u
    R = (x, y, t, u, p) -> 0.0
    u₀ = [f(pts[:, i]...) for i in axes(pts, 2)]
    final_time = 1.0
    prob = FVMProblem(; mesh, iip_flux, boundary_conditions, diffusion=D, reaction=R, initial_condition=u₀, final_time)
    sol = solve(prob, Rosenbrock23(linsolve=KLUFactorization()); jac_prototype=FVM.jacobian_sparsity(prob), saveat=0.01)
    interp = FVM.construct_mesh_interpolant(mesh, sol)
    DelaunayTriangulation.add_ghost_triangles!(T, adj, adj2v, DG)
    return sol, T, adj, adj2v, DG, pts, interp
end
r = 10.0 .^ (LinRange(log10(1.0), log10(0.027), 25))
sol = []
pts, T, adj, adj2v, DG, interp = [], [], [], [], [], []
for _r in r
    _sol, _T, _adj, _adj2v, _DG, _pts, _interp = evaluate_linearly_singular_problem(_r)
    push!(sol, _sol)
    push!(T, _T)
    push!(adj, _adj)
    push!(adj2v, _adj2v)
    push!(DG, _DG)
    push!(pts, _pts)
    push!(interp, _interp)
end
vals = zeros(length(sol[1].t), length(sol))
_i = Ref(1)
trypt = Ref((1, 2, 3))
trypt_exact = Ref((1, 2, 3))
for i in axes(vals, 2)
    _i[] = i
    if i < size(vals, 2)
        vals[:, i] .= sum_error(interp[i], pts[i], adj[i], adj2v[i], DG[i], pts[end], sol[end])
    end
end

########### Quadratic singular diffusion
function evaluate_quadratic_singular_problem(r)
    ## Generate the mesh 
    iip_flux = true
    x₁ = LinRange(0, 2, 1000)
    y₁ = LinRange(0, 0, 1000)
    x₂ = LinRange(2, 2, 1000)
    y₂ = LinRange(0, 2, 1000)
    x₃ = LinRange(2, 0, 1000)
    y₃ = LinRange(2, 2, 1000)
    x₄ = LinRange(0, 0, 1000)
    y₄ = LinRange(2, 0, 1000)
    x = reduce(vcat, [x₁, x₂, x₃, x₄])
    y = reduce(vcat, [y₁, y₂, y₃, y₄])
    xy = [[x[i], y[i]] for i in eachindex(x)]
    unique!(xy)
    x = [xy[i][1] for i in eachindex(xy)]
    y = [xy[i][2] for i in eachindex(xy)]
    T, adj, adj2v, DG, pts, BN = generate_mesh(x, y, r; gmsh_path=GMSH_PATH)
    ## Define the problem 
    mesh = FVMGeometry(T, adj, adj2v, DG, pts, BN)
    boundary_functions = Vector{Function}([(x, y, t, p) -> x < 1.0 ? 5.0 : 10.0])
    type = ["Dirichlet"]
    boundary_conditions = BoundaryConditions(mesh, boundary_functions, type)
    f = (x, y) -> 7.5
    D = (x, y, t, u, p) -> 8.0 / u^2
    R = (x, y, t, u, p) -> 0.0
    u₀ = [f(pts[:, i]...) for i in axes(pts, 2)]
    final_time = 1.0
    prob = FVMProblem(; mesh, iip_flux, boundary_conditions, diffusion=D, reaction=R, initial_condition=u₀, final_time)
    sol = solve(prob, Rosenbrock23(linsolve=KLUFactorization()); jac_prototype=FVM.jacobian_sparsity(prob), saveat=0.01)
    interp = FVM.construct_mesh_interpolant(mesh, sol)
    DelaunayTriangulation.add_ghost_triangles!(T, adj, adj2v, DG)
    return sol, T, adj, adj2v, DG, pts, interp
end
sol = []
pts, T, adj, adj2v, DG, interp = [], [], [], [], [], []
for _r in r
    _sol, _T, _adj, _adj2v, _DG, _pts, _interp = evaluate_quadratic_singular_problem(_r)
    push!(sol, _sol)
    push!(T, _T)
    push!(adj, _adj)
    push!(adj2v, _adj2v)
    push!(DG, _DG)
    push!(pts, _pts)
    push!(interp, _interp)
end
vals2 = zeros(length(sol[1].t), length(sol))
_i = Ref(1)
trypt = Ref((1, 2, 3))
trypt_exact = Ref((1, 2, 3))
for i in axes(vals2, 2)
    _i[] = i
    if i < size(vals2, 2)
        vals2[:, i] .= sum_error(interp[i], pts[i], adj[i], adj2v[i], DG[i], pts[end], sol[end])
    end
end

########### Quadratic singular diffusion + R(u)
function evaluate_quadratic_reaction_singular_problem(r)
    ## Generate the mesh 
    iip_flux = true
    x₁ = LinRange(0, 2, 1000)
    y₁ = LinRange(0, 0, 1000)
    x₂ = LinRange(2, 2, 1000)
    y₂ = LinRange(0, 2, 1000)
    x₃ = LinRange(2, 0, 1000)
    y₃ = LinRange(2, 2, 1000)
    x₄ = LinRange(0, 0, 1000)
    y₄ = LinRange(2, 0, 1000)
    x = reduce(vcat, [x₁, x₂, x₃, x₄])
    y = reduce(vcat, [y₁, y₂, y₃, y₄])
    xy = [[x[i], y[i]] for i in eachindex(x)]
    unique!(xy)
    x = [xy[i][1] for i in eachindex(xy)]
    y = [xy[i][2] for i in eachindex(xy)]
    T, adj, adj2v, DG, pts, BN = generate_mesh(x, y, r; gmsh_path=GMSH_PATH)
    ## Define the problem 
    mesh = FVMGeometry(T, adj, adj2v, DG, pts, BN)
    boundary_functions = Vector{Function}([(x, y, t, p) -> x < 1.0 ? 5.0 : 10.0])
    type = ["Dirichlet"]
    boundary_conditions = BoundaryConditions(mesh, boundary_functions, type)
    f = (x, y) -> 7.5
    D = (x, y, t, u, p) -> 8.0 / u^2
    R = (x, y, t, u, p) -> 0.5u * (1 - -u / 10.0)
    u₀ = [f(pts[:, i]...) for i in axes(pts, 2)]
    final_time = 1.0
    prob = FVMProblem(; mesh, iip_flux, boundary_conditions, diffusion=D, reaction=R, initial_condition=u₀, final_time)
    sol = solve(prob, Rosenbrock23(linsolve=KLUFactorization()); jac_prototype=FVM.jacobian_sparsity(prob), saveat=0.01)
    interp = FVM.construct_mesh_interpolant(mesh, sol)
    DelaunayTriangulation.add_ghost_triangles!(T, adj, adj2v, DG)
    return sol, T, adj, adj2v, DG, pts, interp
end
sol = []
pts, T, adj, adj2v, DG, interp = [], [], [], [], [], []
for _r in r
    _sol, _T, _adj, _adj2v, _DG, _pts, _interp = evaluate_quadratic_reaction_singular_problem(_r)
    push!(sol, _sol)
    push!(T, _T)
    push!(adj, _adj)
    push!(adj2v, _adj2v)
    push!(DG, _DG)
    push!(pts, _pts)
    push!(interp, _interp)
end
vals3 = zeros(length(sol[1].t), length(sol))
_i = Ref(1)
trypt = Ref((1, 2, 3))
trypt_exact = Ref((1, 2, 3))
for i in axes(vals3, 2)
    _i[] = i
    if i < size(vals3, 2)
        vals3[:, i] .= sum_error(interp[i], pts[i], adj[i], adj2v[i], DG[i], pts[end], sol[end])
    end
end

fig = Figure(fontsize=38, resolution=(2331.2319f0, 567.25476f0))
colors = cgrad(:viridis, size.(pts, 2) / size(pts[end], 2); categorical=true)
ax = Axis(fig[1, 1],
    width=600,
    height=400,
    title=L"(a): $D(u) = 8/u$",
    titlealign=:left,
    xlabel=L"t",
    ylabel=L"E(t)",
    xticks=(0:0.5:1, [L"%$s" for s in 0:0.5:1]),
    yticks=(0:0.5:1, [L"%$s" for s in 0:0.5:1]))
for i in axes(vals, 2)
    lines!(ax, sol[i].t, vals[:, i], color=colors[i])
end
ax = Axis(fig[1, 2],
    width=600,
    height=400,
    title=L"(b): $D(u) = 8/u^2$",
    titlealign=:left,
    xlabel=L"t",
    ylabel=L"E(t)",
    xticks=(0:0.5:1, [L"%$s" for s in 0:0.5:1]),
    yticks=(0:0.5:1, [L"%$s" for s in 0:0.5:1]))
for i in axes(vals2, 2)
    lines!(ax, sol[i].t, vals2[:, i], color=colors[i])
end
ax = Axis(fig[1, 3],
    width=600,
    height=400,
    title=L"(c): $D(u) = 8/u^2$, $R(u) = (u/2)(1 - u /10)$",
    titlealign=:left,
    xlabel=L"t",
    ylabel=L"E(t)",
    xticks=(0:0.5:1, [L"%$s" for s in 0:0.5:1]),
    yticks=(0:0.5:1, [L"%$s" for s in 0:0.5:1]))
for i in axes(vals3, 2)
    lines!(ax, sol[i].t, vals3[:, i], color=colors[i])
end
ylims!(ax, 0, 1)
Colorbar(fig[1, 4], limits=(0.0, 6000.0), colormap=colors, label=L"n", vertical=true,
    ticks=(0:1500:6000, [L"%$s" for s in 0:1500:6000]))
save("testing_curves.pdf", fig)



#############
#
#
#
#
#
#
#
#
#

########### Linearly singular diffusion
function evaluate_linearly_singular_problem(r)
    ## Generate the mesh 
    iip_flux = true
    x₁ = LinRange(0, 2, 1000)
    y₁ = LinRange(0, 0, 1000)
    x₂ = LinRange(2, 2, 1000)
    y₂ = LinRange(0, 2, 1000)
    x₃ = LinRange(2, 0, 1000)
    y₃ = LinRange(2, 2, 1000)
    x₄ = LinRange(0, 0, 1000)
    y₄ = LinRange(2, 0, 1000)
    x = reduce(vcat, [x₁, x₂, x₃, x₄])
    y = reduce(vcat, [y₁, y₂, y₃, y₄])
    xy = [[x[i], y[i]] for i in eachindex(x)]
    unique!(xy)
    x = [xy[i][1] for i in eachindex(xy)]
    y = [xy[i][2] for i in eachindex(xy)]
    T, adj, adj2v, DG, pts, BN = generate_mesh(x, y, r; gmsh_path=GMSH_PATH)
    ## Define the problem 
    mesh = FVMGeometry(T, adj, adj2v, DG, pts, BN)
    boundary_functions = Vector{Function}([(x, y, t, u, p) -> 0.5 * u * (1 - u / 10.0)])
    type = ["dudt"]
    boundary_conditions = BoundaryConditions(mesh, boundary_functions, type)
    f = (x, y) -> 7.5
    D = (x, y, t, u, p) -> 8.0 / u
    R = (x, y, t, u, p) -> 0.0
    u₀ = [f(pts[:, i]...) for i in axes(pts, 2)]
    final_time = 1.0
    prob = FVMProblem(; mesh, iip_flux, boundary_conditions, diffusion=D, reaction=R, initial_condition=u₀, final_time)
    sol = solve(prob, Rosenbrock23(linsolve=KLUFactorization()); jac_prototype=FVM.jacobian_sparsity(prob), saveat=0.01)
    interp = FVM.construct_mesh_interpolant(mesh, sol)
    DelaunayTriangulation.add_ghost_triangles!(T, adj, adj2v, DG)
    return sol, T, adj, adj2v, DG, pts, interp
end
r = 10.0 .^ (LinRange(log10(1.0), log10(0.027), 25))
sol = []
pts, T, adj, adj2v, DG, interp = [], [], [], [], [], []
for _r in r
    _sol, _T, _adj, _adj2v, _DG, _pts, _interp = evaluate_linearly_singular_problem(_r)
    push!(sol, _sol)
    push!(T, _T)
    push!(adj, _adj)
    push!(adj2v, _adj2v)
    push!(DG, _DG)
    push!(pts, _pts)
    push!(interp, _interp)
end
vals = zeros(length(sol[1].t), length(sol))
_i = Ref(1)
trypt = Ref((1, 2, 3))
trypt_exact = Ref((1, 2, 3))
for i in axes(vals, 2)
    _i[] = i
    if i < size(vals, 2)
        vals[:, i] .= sum_error(interp[i], pts[i], adj[i], adj2v[i], DG[i], pts[end], sol[end])
    end
end

########### Quadratic singular diffusion
function evaluate_quadratic_singular_problem(r)
    ## Generate the mesh 
    iip_flux = true
    x₁ = LinRange(0, 2, 1000)
    y₁ = LinRange(0, 0, 1000)
    x₂ = LinRange(2, 2, 1000)
    y₂ = LinRange(0, 2, 1000)
    x₃ = LinRange(2, 0, 1000)
    y₃ = LinRange(2, 2, 1000)
    x₄ = LinRange(0, 0, 1000)
    y₄ = LinRange(2, 0, 1000)
    x = reduce(vcat, [x₁, x₂, x₃, x₄])
    y = reduce(vcat, [y₁, y₂, y₃, y₄])
    xy = [[x[i], y[i]] for i in eachindex(x)]
    unique!(xy)
    x = [xy[i][1] for i in eachindex(xy)]
    y = [xy[i][2] for i in eachindex(xy)]
    T, adj, adj2v, DG, pts, BN = generate_mesh(x, y, r; gmsh_path=GMSH_PATH)
    ## Define the problem 
    mesh = FVMGeometry(T, adj, adj2v, DG, pts, BN)
    boundary_functions = Vector{Function}([(x, y, t, u, p) -> 0.5 * u * (1 - u / 10.0)])
    type = ["dudt"]
    boundary_conditions = BoundaryConditions(mesh, boundary_functions, type)
    f = (x, y) -> 7.5
    D = (x, y, t, u, p) -> 8.0 / u^2
    R = (x, y, t, u, p) -> 0.0
    u₀ = [f(pts[:, i]...) for i in axes(pts, 2)]
    final_time = 1.0
    prob = FVMProblem(; mesh, iip_flux, boundary_conditions, diffusion=D, reaction=R, initial_condition=u₀, final_time)
    sol = solve(prob, Rosenbrock23(linsolve=KLUFactorization()); jac_prototype=FVM.jacobian_sparsity(prob), saveat=0.01)
    interp = FVM.construct_mesh_interpolant(mesh, sol)
    DelaunayTriangulation.add_ghost_triangles!(T, adj, adj2v, DG)
    return sol, T, adj, adj2v, DG, pts, interp
end
sol = []
pts, T, adj, adj2v, DG, interp = [], [], [], [], [], []
for _r in r
    _sol, _T, _adj, _adj2v, _DG, _pts, _interp = evaluate_quadratic_singular_problem(_r)
    push!(sol, _sol)
    push!(T, _T)
    push!(adj, _adj)
    push!(adj2v, _adj2v)
    push!(DG, _DG)
    push!(pts, _pts)
    push!(interp, _interp)
end
vals2 = zeros(length(sol[1].t), length(sol))
_i = Ref(1)
trypt = Ref((1, 2, 3))
trypt_exact = Ref((1, 2, 3))
for i in axes(vals2, 2)
    _i[] = i
    if i < size(vals2, 2)
        vals2[:, i] .= sum_error(interp[i], pts[i], adj[i], adj2v[i], DG[i], pts[end], sol[end])
    end
end

########### Quadratic singular diffusion + R(u)
function evaluate_quadratic_reaction_singular_problem(r)
    ## Generate the mesh 
    iip_flux = true
    x₁ = LinRange(0, 2, 1000)
    y₁ = LinRange(0, 0, 1000)
    x₂ = LinRange(2, 2, 1000)
    y₂ = LinRange(0, 2, 1000)
    x₃ = LinRange(2, 0, 1000)
    y₃ = LinRange(2, 2, 1000)
    x₄ = LinRange(0, 0, 1000)
    y₄ = LinRange(2, 0, 1000)
    x = reduce(vcat, [x₁, x₂, x₃, x₄])
    y = reduce(vcat, [y₁, y₂, y₃, y₄])
    xy = [[x[i], y[i]] for i in eachindex(x)]
    unique!(xy)
    x = [xy[i][1] for i in eachindex(xy)]
    y = [xy[i][2] for i in eachindex(xy)]
    T, adj, adj2v, DG, pts, BN = generate_mesh(x, y, r; gmsh_path=GMSH_PATH)
    ## Define the problem 
    mesh = FVMGeometry(T, adj, adj2v, DG, pts, BN)
    boundary_functions = Vector{Function}([(x, y, t, u, p) -> 0.5 * u * (1 - u / 10.0)])
    type = ["dudt"]
    boundary_conditions = BoundaryConditions(mesh, boundary_functions, type)
    f = (x, y) -> 7.5
    D = (x, y, t, u, p) -> 8.0 / u^2
    R = (x, y, t, u, p) -> 0.5u * (1 - -u / 10.0)
    u₀ = [f(pts[:, i]...) for i in axes(pts, 2)]
    final_time = 1.0
    prob = FVMProblem(; mesh, iip_flux, boundary_conditions, diffusion=D, reaction=R, initial_condition=u₀, final_time)
    sol = solve(prob, Rosenbrock23(linsolve=KLUFactorization()); jac_prototype=FVM.jacobian_sparsity(prob), saveat=0.01)
    interp = FVM.construct_mesh_interpolant(mesh, sol)
    DelaunayTriangulation.add_ghost_triangles!(T, adj, adj2v, DG)
    return sol, T, adj, adj2v, DG, pts, interp
end
sol = []
pts, T, adj, adj2v, DG, interp = [], [], [], [], [], []
for _r in r
    _sol, _T, _adj, _adj2v, _DG, _pts, _interp = evaluate_quadratic_reaction_singular_problem(_r)
    push!(sol, _sol)
    push!(T, _T)
    push!(adj, _adj)
    push!(adj2v, _adj2v)
    push!(DG, _DG)
    push!(pts, _pts)
    push!(interp, _interp)
end
vals3 = zeros(length(sol[1].t), length(sol))
_i = Ref(1)
trypt = Ref((1, 2, 3))
trypt_exact = Ref((1, 2, 3))
for i in axes(vals3, 2)
    _i[] = i
    if i < size(vals3, 2)
        vals3[:, i] .= sum_error(interp[i], pts[i], adj[i], adj2v[i], DG[i], pts[end], sol[end])
    end
end

fig = Figure(fontsize=38, resolution=(2453.848f0, 567.25476f0))
colors = cgrad(:viridis, size.(pts, 2) / size(pts[end], 2); categorical=true)
ax = Axis(fig[1, 1],
    width=600,
    height=400,
    title=L"(a): $D(u) = 8/u$",
    titlealign=:left,
    xlabel=L"t",
    ylabel=L"E(t)",
    xticks=(0:0.5:1, [L"%$s" for s in 0:0.5:1]),
    yticks=(0:0.0002:0.0008, [L"%$s" for s in 0:0.0002:0.0008]))
for i in axes(vals, 2)
    lines!(ax, sol[i].t, vals[:, i], color=colors[i])
end
ax = Axis(fig[1, 2],
    width=600,
    height=400,
    title=L"(b): $D(u) = 8/u^2$",
    titlealign=:left,
    xlabel=L"t",
    ylabel=L"E(t)",
    xticks=(0:0.5:1, [L"%$s" for s in 0:0.5:1]),
    yticks=(0:0.005:0.025, [L"%$s" for s in 0:0.005:0.025]))
for i in axes(vals2, 2)
    lines!(ax, sol[i].t, vals2[:, i], color=colors[i])
end
ylims!(ax, 0, 0.025)
ax = Axis(fig[1, 3],
    width=600,
    height=400,
    title=L"(c): $D(u) = 8/u^2$, $R(u) = (u/2)(1 - u /10)$",
    titlealign=:left,
    xlabel=L"t",
    ylabel=L"E(t)",
    xticks=(0:0.5:1, [L"%$s" for s in 0:0.5:1]),
    yticks=(0:0.5:1, [L"%$s" for s in 0:0.5:1]))
for i in axes(vals3, 2)
    lines!(ax, sol[i].t, vals3[:, i], color=colors[i])
end
ylims!(ax, 0, 1)
Colorbar(fig[1, 4], limits=(0.0, 6000.0), colormap=colors, label=L"n", vertical=true,
    ticks=(0:1500:6000, [L"%$s" for s in 0:1500:6000]))
save("testing_curves_2.pdf", fig)

############################################################################
## POROUS-MEDIUM
############################################################################
function approx_δ(x, y, ε)
    return 1 / (ε^2 * π) * exp(-(x^2 + y^2) / ε^2)
end
function eval_RmM(m, M)
    return 4m / (m - 1) * (M / (4π))^((m - 1) / m)
end
function eval_exact_solution(x, y, t, m, M)
    normx2 = x^2 + y^2
    RmM = eval_RmM(m, M)
    if normx2 < RmM * t^(1 / m)
        return t^(-1 / m) * ((M / (4π))^((m - 1) / m) - (m - 1) / (4m) * normx2 * t^(-1 / m))^(1 / (m - 1))
    else
        return 0.0
    end
end
final_time = 12.0
m = 2
M = 0.37
RmM = eval_RmM(m, M)
Diffus = 2.53
L = RmM^(1 / 2) * (Diffus * final_time)^(1 / (2m))
a = -L
b = L
c = -L
d = L
T, adj, adj2v, DG, points, BN = triangulate_structured(a, b, c, d, 25, 25; return_boundary_types=true, single_boundary=true)
T, adj, adj2v, DG, points, BN = generate_mesh(points[1, BN], points[2, BN], 0.1; gmsh_path=GMSH_PATH)
mesh = FVMGeometry(T, adj, adj2v, DG, points, BN)
type = [:D]
boundary_conditions = BoundaryConditions(mesh, Vector{Function}([(x, y, t, p) -> 0.0]), type)
ε = 1e-1
f = (x, y) -> M * approx_δ(x, y, ε)
D = (x, y, t, u, p) -> Diffus * u^(m - 1)
R = (x, y, t, u, p) -> 0.0
u₀ = f.(points[1, :], points[2, :])
iip_flux = true
prob = FVMProblem(; mesh, iip_flux, boundary_conditions, diffusion_parameters=Diffus, diffusion=D, reaction=R, initial_condition=u₀, final_time)
sol = solve(prob, Rosenbrock23(linsolve=KLUFactorization()); saveat=[0.0, 4.0, 8.0, 12.0])

fig = Figure(fontsize=46, resolution=(2721.39f0, 2346.4521f0))
ax1 = Axis(fig[1, 1], xlabel=L"x", ylabel=L"y", title=L"(a): Numerical solution, $t = 4$", titlealign=:left, width=600, height=600, aspect=1, xticks=(-2.5:2.5, [L"%$s" for s in -2.5:2.5]), yticks=(-2.5:2.5, [L"%$s" for s in -2.5:2.5]))
ax2 = Axis(fig[1, 2], xlabel=L"x", ylabel=L"y", title=L"(b): Exact solution, $t = 4$", titlealign=:left, width=600, height=600, aspect=1, xticks=(-2.5:2.5, [L"%$s" for s in -2.5:2.5]), yticks=(-2.5:2.5, [L"%$s" for s in -2.5:2.5]))
ax3 = Axis(fig[1, 4], xlabel=L"x", ylabel=L"y", title=L"(c): Error, $t = 4$", titlealign=:left, width=600, height=600, aspect=1, xticks=(-2.5:2.5, [L"%$s" for s in -2.5:2.5]), yticks=(-2.5:2.5, [L"%$s" for s in -2.5:2.5]))
ax4 = Axis(fig[2, 1], xlabel=L"x", ylabel=L"y", title=L"(d): Numerical solution, $t = 8$", titlealign=:left, width=600, height=600, aspect=1, xticks=(-2.5:2.5, [L"%$s" for s in -2.5:2.5]), yticks=(-2.5:2.5, [L"%$s" for s in -2.5:2.5]))
ax5 = Axis(fig[2, 2], xlabel=L"x", ylabel=L"y", title=L"(e): Exact solution, $t = 8$", titlealign=:left, width=600, height=600, aspect=1, xticks=(-2.5:2.5, [L"%$s" for s in -2.5:2.5]), yticks=(-2.5:2.5, [L"%$s" for s in -2.5:2.5]))
ax6 = Axis(fig[2, 4], xlabel=L"x", ylabel=L"y", title=L"(f): Error, $t = 8$", titlealign=:left, width=600, height=600, aspect=1, xticks=(-2.5:2.5, [L"%$s" for s in -2.5:2.5]), yticks=(-2.5:2.5, [L"%$s" for s in -2.5:2.5]))
ax7 = Axis(fig[3, 1], xlabel=L"x", ylabel=L"y", title=L"(g): Numerical solution, $t = 12$", titlealign=:left, width=600, height=600, aspect=1, xticks=(-2.5:2.5, [L"%$s" for s in -2.5:2.5]), yticks=(-2.5:2.5, [L"%$s" for s in -2.5:2.5]))
ax8 = Axis(fig[3, 2], xlabel=L"x", ylabel=L"y", title=L"(h): Exact solution, $t = 12$", titlealign=:left, width=600, height=600, aspect=1, xticks=(-2.5:2.5, [L"%$s" for s in -2.5:2.5]), yticks=(-2.5:2.5, [L"%$s" for s in -2.5:2.5]))
ax9 = Axis(fig[3, 4], xlabel=L"x", ylabel=L"y", title=L"(i): Error, $t = 12$", titlealign=:left, width=600, height=600, aspect=1, xticks=(-2.5:2.5, [L"%$s" for s in -2.5:2.5]), yticks=(-2.5:2.5, [L"%$s" for s in -2.5:2.5]))

mesh!(ax1, Matrix(points'), [collect(T)[i][j] for i in 1:length(T), j in 1:3], color=sol.u[2], colormap=:matter, colorrange=(0, 0.1 / 2))
mesh!(ax4, Matrix(points'), [collect(T)[i][j] for i in 1:length(T), j in 1:3], color=sol.u[3], colormap=:matter, colorrange=(0, 0.1 / 2))
mesh!(ax7, Matrix(points'), [collect(T)[i][j] for i in 1:length(T), j in 1:3], color=sol.u[4], colormap=:matter, colorrange=(0, 0.1 / 2))

ue1 = eval_exact_solution.(points[1, :], points[2, :], sol.t[2] * Diffus, m, M)
ue2 = eval_exact_solution.(points[1, :], points[2, :], sol.t[3] * Diffus, m, M)
ue3 = eval_exact_solution.(points[1, :], points[2, :], sol.t[4] * Diffus, m, M)

mesh!(ax2, Matrix(points'), [collect(T)[i][j] for i in 1:length(T), j in 1:3], color=ue1, colormap=:matter, colorrange=(0, 0.1 / 2))
mesh!(ax5, Matrix(points'), [collect(T)[i][j] for i in 1:length(T), j in 1:3], color=ue2, colormap=:matter, colorrange=(0, 0.1 / 2))
mesh!(ax8, Matrix(points'), [collect(T)[i][j] for i in 1:length(T), j in 1:3], color=ue3, colormap=:matter, colorrange=(0, 0.1 / 2))

Colorbar(fig[1:3, 3], limits=(0.0, 0.1 / 2), colormap=:matter, label=L"u(x, y, t)", vertical=true, ticks=(0:0.01:0.05, [L"%$s" for s in 0:0.01:0.05]))

er1 = 100 * abs.(sol.u[2] .- ue1) ./ eval_exact_solution(0.0, 0.0, sol.t[2] * Diffus, m, M)
er2 = 100 * abs.(sol.u[3] .- ue2) ./ eval_exact_solution(0.0, 0.0, sol.t[3] * Diffus, m, M)
er3 = 100 * abs.(sol.u[4] .- ue3) ./ eval_exact_solution(0.0, 0.0, sol.t[4] * Diffus, m, M)

mesh!(ax3, Matrix(points'), [collect(T)[i][j] for i in 1:length(T), j in 1:3], color=er1, colormap=:matter, colorrange=(0, 1))
mesh!(ax6, Matrix(points'), [collect(T)[i][j] for i in 1:length(T), j in 1:3], color=er2, colormap=:matter, colorrange=(0, 1))
mesh!(ax9, Matrix(points'), [collect(T)[i][j] for i in 1:length(T), j in 1:3], color=er3, colormap=:matter, colorrange=(0, 1))

Colorbar(fig[1:3, 5], limits=(0.0, 1), colormap=:matter, label=L"e(x, y, t) (%)", vertical=true, ticks=(0:0.2:1, [L"%$s" for s in 0:0.2:1]))

R1 = sqrt(RmM * (Diffus * sol.t[2])^(1 / m))
R2 = sqrt(RmM * (Diffus * sol.t[3])^(1 / m))
R3 = sqrt(RmM * (Diffus * sol.t[4])^(1 / m))

θ = LinRange(0, 2π, 2000)
ℓ1 = R1 .* exp.(im * θ)
ℓ2 = R2 .* exp.(im * θ)
ℓ3 = R3 .* exp.(im * θ)

[lines!(ax, real(ℓ1), imag(ℓ1), color=:black, linewidth=4) for ax in (ax1, ax2, ax3)]
[lines!(ax, real(ℓ2), imag(ℓ2), color=:black, linewidth=4) for ax in (ax4, ax5, ax6)]
[lines!(ax, real(ℓ3), imag(ℓ3), color=:black, linewidth=4) for ax in (ax7, ax8, ax9)]

fig

save("testing_maps_heat_1.png", fig)

############################################################################
## POROUS-MEDIUM WITH LINEAR SOURCE
############################################################################
function approx_δ(x, y, ε)
    return 1 / (ε^2 * π) * exp(-(x^2 + y^2) / ε^2)
end
function eval_RmM(m, M)
    return 4m / (m - 1) * (M / (4π))^((m - 1) / m)
end
function eval_exact_solution(x, y, t, m, M)
    normx2 = x^2 + y^2
    RmM = eval_RmM(m, M)
    if normx2 < RmM * t^(1 / m)
        return t^(-1 / m) * ((M / (4π))^((m - 1) / m) - (m - 1) / (4m) * normx2 * t^(-1 / m))^(1 / (m - 1))
    else
        return 0.0
    end
end
function eval_exact_solution(x, y, t, m, M, D, λ)
    return exp(λ * t) * eval_exact_solution(x, y, D / (λ * (m - 1)) * (exp(λ * (m - 1) * t) - 1), m, M)
end
final_time = 3.0
m = 2
M = 5.3
RmM = eval_RmM(m, M)
Diffus = 7.3
Lambda = 0.2
L = RmM^(1 / 2) * (Diffus / (Lambda * (1 - m)) + Diffus / (Lambda * (m - 1)) * exp(Lambda * (m - 1) * final_time))^(1 / (2m))
a = -L
b = L
c = -L
d = L
T, adj, adj2v, DG, points, BN = triangulate_structured(a, b, c, d, 25, 25; return_boundary_types=true, single_boundary=true)
T, adj, adj2v, DG, points, BN = generate_mesh(points[1, BN], points[2, BN], 0.15; gmsh_path=GMSH_PATH)
mesh = FVMGeometry(T, adj, adj2v, DG, points, BN)
type = [:D]
boundary_conditions = BoundaryConditions(mesh, Vector{Function}([(x, y, t, p) -> 0.0]), type)
ε = 1e-1
f = (x, y) -> M * approx_δ(x, y, ε)
D = (x, y, t, u, p) -> p[1] * u^(p[2] - 1)
R = (x, y, t, u, p) -> p[1] * u
u₀ = f.(points[1, :], points[2, :])
iip_flux = true
prob = FVMProblem(; mesh, iip_flux, boundary_conditions, diffusion_parameters=(Diffus, m), reaction_parameters=Lambda, diffusion=D, reaction=R, initial_condition=u₀, final_time)
sol = solve(prob, Rosenbrock23(linsolve=KLUFactorization()); saveat=[0.0, 1.0, 2.0, 3.0])

fig = Figure(fontsize=46, resolution=(2721.39f0, 2346.4521f0))
ax1 = Axis(fig[1, 1], xlabel=L"x", ylabel=L"y", title=L"(a): Numerical solution, $t = 1$", titlealign=:left, width=600, height=600, aspect=1, xticks=(-6:2:6, [L"%$s" for s in -6:2:6]), yticks=(-6:2:6, [L"%$s" for s in -6:2:6]))
ax2 = Axis(fig[1, 2], xlabel=L"x", ylabel=L"y", title=L"(b): Exact solution, $t = 1$", titlealign=:left, width=600, height=600, aspect=1, xticks=(-6:2:6, [L"%$s" for s in -6:2:6]), yticks=(-6:2:6, [L"%$s" for s in -6:2:6]))
ax3 = Axis(fig[1, 4], xlabel=L"x", ylabel=L"y", title=L"(c): Error, $t = 1$", titlealign=:left, width=600, height=600, aspect=1, xticks=(-6:2:6, [L"%$s" for s in -6:2:6]), yticks=(-6:2:6, [L"%$s" for s in -6:2:6]))
ax4 = Axis(fig[2, 1], xlabel=L"x", ylabel=L"y", title=L"(d): Numerical solution, $t = 2$", titlealign=:left, width=600, height=600, aspect=1, xticks=(-6:2:6, [L"%$s" for s in -6:2:6]), yticks=(-6:2:6, [L"%$s" for s in -6:2:6]))
ax5 = Axis(fig[2, 2], xlabel=L"x", ylabel=L"y", title=L"(e): Exact solution, $t = 2$", titlealign=:left, width=600, height=600, aspect=1, xticks=(-6:2:6, [L"%$s" for s in -6:2:6]), yticks=(-6:2:6, [L"%$s" for s in -6:2:6]))
ax6 = Axis(fig[2, 4], xlabel=L"x", ylabel=L"y", title=L"(f): Error, $t = 2$", titlealign=:left, width=600, height=600, aspect=1, xticks=(-6:2:6, [L"%$s" for s in -6:2:6]), yticks=(-6:2:6, [L"%$s" for s in -6:2:6]))
ax7 = Axis(fig[3, 1], xlabel=L"x", ylabel=L"y", title=L"(g): Numerical solution, $t = 3$", titlealign=:left, width=600, height=600, aspect=1, xticks=(-6:2:6, [L"%$s" for s in -6:2:6]), yticks=(-6:2:6, [L"%$s" for s in -6:2:6]))
ax8 = Axis(fig[3, 2], xlabel=L"x", ylabel=L"y", title=L"(h): Exact solution, $t = 3$", titlealign=:left, width=600, height=600, aspect=1, xticks=(-6:2:6, [L"%$s" for s in -6:2:6]), yticks=(-6:2:6, [L"%$s" for s in -6:2:6]))
ax9 = Axis(fig[3, 4], xlabel=L"x", ylabel=L"y", title=L"(i): Error, $t = 3$", titlealign=:left, width=600, height=600, aspect=1, xticks=(-6:2:6, [L"%$s" for s in -6:2:6]), yticks=(-6:2:6, [L"%$s" for s in -6:2:6]))

mesh!(ax1, Matrix(points'), [collect(T)[i][j] for i in 1:length(T), j in 1:3], color=sol.u[2], colormap=:matter, colorrange=(0, 1 / 2))
mesh!(ax4, Matrix(points'), [collect(T)[i][j] for i in 1:length(T), j in 1:3], color=sol.u[3], colormap=:matter, colorrange=(0, 1 / 2))
mesh!(ax7, Matrix(points'), [collect(T)[i][j] for i in 1:length(T), j in 1:3], color=sol.u[4], colormap=:matter, colorrange=(0, 1 / 2))

ue1 = eval_exact_solution.(points[1, :], points[2, :], sol.t[2], m, M, Diffus, Lambda)
ue2 = eval_exact_solution.(points[1, :], points[2, :], sol.t[3], m, M, Diffus, Lambda)
ue3 = eval_exact_solution.(points[1, :], points[2, :], sol.t[4], m, M, Diffus, Lambda)

mesh!(ax2, Matrix(points'), [collect(T)[i][j] for i in 1:length(T), j in 1:3], color=ue1, colormap=:matter, colorrange=(0, 1 / 2))
mesh!(ax5, Matrix(points'), [collect(T)[i][j] for i in 1:length(T), j in 1:3], color=ue2, colormap=:matter, colorrange=(0, 1 / 2))
mesh!(ax8, Matrix(points'), [collect(T)[i][j] for i in 1:length(T), j in 1:3], color=ue3, colormap=:matter, colorrange=(0, 1 / 2))

Colorbar(fig[1:3, 3], limits=(0.0, 0.5), colormap=:matter, label=L"u(x, y, t)", vertical=true, ticks=(0:0.1:0.5, [L"%$s" for s in 0:0.1:0.5]))

er1 = 100 * abs.(sol.u[2] .- ue1) ./ eval_exact_solution(0.0, 0.0, sol.t[2] * Diffus, m, M, Diffus, Lambda)
er2 = 100 * abs.(sol.u[3] .- ue2) ./ eval_exact_solution(0.0, 0.0, sol.t[3] * Diffus, m, M, Diffus, Lambda)
er3 = 100 * abs.(sol.u[4] .- ue3) ./ eval_exact_solution(0.0, 0.0, sol.t[4] * Diffus, m, M, Diffus, Lambda)

mesh!(ax3, Matrix(points'), [collect(T)[i][j] for i in 1:length(T), j in 1:3], color=er1, colormap=:matter, colorrange=(0, 1))
mesh!(ax6, Matrix(points'), [collect(T)[i][j] for i in 1:length(T), j in 1:3], color=er2, colormap=:matter, colorrange=(0, 1))
mesh!(ax9, Matrix(points'), [collect(T)[i][j] for i in 1:length(T), j in 1:3], color=er3, colormap=:matter, colorrange=(0, 1))

Colorbar(fig[1:3, 5], limits=(0.0, 1), colormap=:matter, label=L"e(x, y, t) (%)", vertical=true, ticks=(0:0.2:1, [L"%$s" for s in 0:0.2:1]))

R1 = sqrt(RmM * (Diffus / (Lambda * (1 - m)) + Diffus / (Lambda * (m - 1)) * exp(Lambda * (m - 1) * sol.t[2]))^(1 / m))
R2 = sqrt(RmM * (Diffus / (Lambda * (1 - m)) + Diffus / (Lambda * (m - 1)) * exp(Lambda * (m - 1) * sol.t[3]))^(1 / m))
R3 = sqrt(RmM * (Diffus / (Lambda * (1 - m)) + Diffus / (Lambda * (m - 1)) * exp(Lambda * (m - 1) * sol.t[4]))^(1 / m))

θ = LinRange(0, 2π, 2000)
ℓ1 = R1 .* exp.(im * θ)
ℓ2 = R2 .* exp.(im * θ)
ℓ3 = R3 .* exp.(im * θ)

[lines!(ax, real(ℓ1), imag(ℓ1), color=:black, linewidth=4) for ax in (ax1, ax2, ax3)]
[lines!(ax, real(ℓ2), imag(ℓ2), color=:black, linewidth=4) for ax in (ax4, ax5, ax6)]
[lines!(ax, real(ℓ3), imag(ℓ3), color=:black, linewidth=4) for ax in (ax7, ax8, ax9)]

fig

save("testing_maps_heat_source_1.png", fig)