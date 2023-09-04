module FiniteVolumeMethod

using DelaunayTriangulation
using FunctionWrappersWrappers
using PreallocationTools
using LinearAlgebra
using SparseArrays
using SciMLBase
using Base.Threads
using DiffEqBase
using ChunkSplitters
using CommonSolve

include("geometry.jl")
include("conditions.jl")
include("problem.jl")
include("equations.jl")
include("solve.jl")

export FVMGeometry,
    FVMProblem,
    FVMSystem,
    SteadyFVMProblem,
    ConstrainedFVMProblem,
    BoundaryConditions,
    InternalConditions,
    Neumann,
    Dudt,
    Dirichlet,
    Constrained,
    solve

end
