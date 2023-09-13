module FiniteVolumeMethod

using DelaunayTriangulation
using PreallocationTools
using LinearAlgebra
using SparseArrays
using SciMLBase
using Base.Threads
using ChunkSplitters
using CommonSolve

include("geometry.jl")
include("conditions.jl")
include("problem.jl")
include("equations/boundary_edge_contributions.jl")
include("equations/control_volumes.jl")
include("equations/dirichlet.jl")
include("equations/individual_flux_contributions.jl")
include("equations/main_equations.jl")
include("equations/shape_functions.jl")
include("equations/source_contributions.jl")
include("equations/triangle_contributions.jl")
include("equations/volume_normalisation.jl")
include("solve.jl")
include("utils.jl")

export FVMGeometry,
    FVMProblem,
    FVMSystem,
    SteadyFVMProblem,
    BoundaryConditions,
    InternalConditions,
    Neumann,
    Dudt,
    Dirichlet,
    Constrained,
    solve,
    compute_flux,
    pl_interpolate

using PrecompileTools
@setup_workload begin
    @compile_workload begin
        # Compile a non-steady problem 
        n = 5
        α = π / 4
        x₁ = [0.0, 1.0]
        y₁ = [0.0, 0.0]
        r₂ = fill(1, n)
        θ₂ = LinRange(0, α, n)
        x₂ = @. r₂ * cos(θ₂)
        y₂ = @. r₂ * sin(θ₂)
        x₃ = [cos(α), 0.0]
        y₃ = [sin(α), 0.0]
        x = [x₁, x₂, x₃]
        y = [y₁, y₂, y₃]
        boundary_nodes, points = convert_boundary_points_to_indices(x, y)
        tri = triangulate(points; boundary_nodes)
        A = get_total_area(tri)
        refine!(tri)
        mesh = FVMGeometry(tri)
        lower_bc = arc_bc = upper_bc = (x, y, t, u, p) -> zero(u)
        types = (Neumann, Dirichlet, Neumann)
        BCs = BoundaryConditions(mesh, (lower_bc, arc_bc, upper_bc), types)
        f = (x, y) -> 1 - sqrt(x^2 + y^2)
        D = (x, y, t, u, p) -> one(u)
        initial_condition = [f(x, y) for (x, y) in each_point(tri)]
        final_time = 0.1
        prob = FVMProblem(mesh, BCs; diffusion_function=D, initial_condition, final_time)
        ode_prob = ODEProblem(prob)
        steady_prob = SteadyFVMProblem(prob)
        nl_prob = SteadyStateProblem(steady_prob)

        # Compile a system
        tri = triangulate_rectangle(0, 100, 0, 100, 5, 5, single_boundary=true)
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
        prob = FVMSystem(u_prob, v_prob)
        ode_prob = ODEProblem(prob)
        steady_prob = SteadyFVMProblem(prob)
        nl_prob = SteadyStateProblem(steady_prob)
    end
end
end
