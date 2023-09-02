module FiniteVolumeMethod

using DelaunayTriangulation
using FunctionWrappersWrappers
using PreallocationTools
using LinearAlgebra
using SparseArrays
using SciMLBase
using Base.Threads
using DiffEqBase
import DiffEqBase: dualgen
using ChunkSplitters

using DelaunayTriangulation: number_type

include("geometry.jl")
include("boundary_conditions.jl")
include("problem.jl")
include("equations.jl")
include("parallel_equations.jl")
include("solve.jl")

export FVMGeometry
export BoundaryConditions
export FVMProblem
export jacobian_sparsity

end
