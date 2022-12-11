module FiniteVolumeMethod

using DelaunayTriangulation
using FunctionWrappersWrappers
using PreallocationTools
using StaticArraysCore
using LinearAlgebra
using SparseArrays
using SciMLBase
using Base.Threads
using DiffEqBase
using FLoops
using MuladdMacro
using LoopVectorization
import DiffEqBase: dualgen
using DelaunayTriangulation: indices

include("geometry.jl")
include("boundary_conditions.jl")
include("problem.jl")
include("equations.jl")
include("parallel_equations.jl")
include("solve.jl")
include("interpolant.jl")

export FVMGeometry
export BoundaryConditions
export FVMProblem
export eval_interpolant
export eval_interpolant!

end
