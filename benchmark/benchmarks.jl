using FiniteVolumeMethod, BenchmarkTools
using DelaunayTriangulation, OrdinaryDiffEqSDIRK, LinearSolve, StableRNGs
using LinearAlgebra

const SUITE = BenchmarkGroup()
const rng = StableRNG(123)
const FVM = FiniteVolumeMethod

# =============================================================================
# Mesh generation and geometry
# =============================================================================

SUITE["mesh"] = BenchmarkGroup()

SUITE["mesh"]["triangulate_rectangle"] = @benchmarkable triangulate_rectangle(
    0.0, 2.0, 0.0, 5.0, 12, 19; single_boundary = false
)

tri = triangulate_rectangle(0.0, 2.0, 0.0, 5.0, 12, 19; single_boundary = false)

SUITE["mesh"]["geometry"] = @benchmarkable FVMGeometry($tri)

mesh = FVMGeometry(tri)

# =============================================================================
# Boundary conditions + problem construction
# =============================================================================

f1 = (x, y, t, u, p) -> x * y + u - p
f2 = (x, y, t, u, p) -> u + p - t
f3 = (x, y, t, u, p) -> x
f4 = (x, y, t, u, p) -> y - x
bc_funs = (f1, f2, f3, f4)
bc_types = (Dirichlet, Neumann, Dirichlet, Neumann)
bc_params = (0.5, 0.2, 0.3, 0.4)

SUITE["bcs"] = BenchmarkGroup()

SUITE["bcs"]["construct"] = @benchmarkable BoundaryConditions(
    $mesh, $bc_funs, $bc_types; parameters = $bc_params
)

BCs = BoundaryConditions(mesh, bc_funs, bc_types; parameters = bc_params)

internal_dirichlet_nodes = Dict([7 + (i - 1) * 12 for i in 2:18] .=> 1)
ICs = InternalConditions(
    (x, y, t, u, p) -> x + y + t + u + p;
    dirichlet_nodes = internal_dirichlet_nodes, parameters = 0.29
)

flux_function = (x, y, t, α, β, γ, p) -> let u = α[1] * x + β[1] * y + γ[1]
    (-α[1] * u * p[1] + t, x + t - β[1] * u * p[2])
end
flux_parameters = (-0.5, 1.3)
source_function = (x, y, t, u, p) -> u + p
source_parameters = 1.5
initial_condition = rand(rng, DelaunayTriangulation.num_solid_vertices(tri))

SUITE["problem"] = BenchmarkGroup()

SUITE["problem"]["construct"] = @benchmarkable FVMProblem(
    $mesh, $BCs, $ICs; flux_function = $flux_function,
    flux_parameters = $flux_parameters, source_function = $source_function,
    source_parameters = $source_parameters,
    initial_condition = $initial_condition,
    initial_time = 2.0, final_time = 5.0
)

prob = FVMProblem(
    mesh, BCs, ICs; flux_function = flux_function,
    flux_parameters = flux_parameters, source_function = source_function,
    source_parameters = source_parameters,
    initial_condition = initial_condition,
    initial_time = 2.0, final_time = 5.0
)

SUITE["problem"]["steady"] = @benchmarkable SteadyFVMProblem($prob)

# =============================================================================
# Solve
# =============================================================================

SUITE["solve"] = BenchmarkGroup()

SUITE["solve"]["trbdf2"] = @benchmarkable solve(
    $prob, TRBDF2(; linsolve = KLUFactorization()); saveat = 0.5
)
