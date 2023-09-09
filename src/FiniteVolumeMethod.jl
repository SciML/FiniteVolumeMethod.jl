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
include("dae.jl")
include("solve.jl")
include("utils.jl")

export FVMGeometry,
    FVMProblem,
    FVMSystem,
    SteadyFVMProblem,
    FVMDAEProblem,
    BoundaryConditions,
    InternalConditions,
    Neumann,
    Dudt,
    Dirichlet,
    Constrained,
    solve,
    compute_flux,
    pl_interpolate,
    get_dae_initial_condition,
    get_differential_vars

end
