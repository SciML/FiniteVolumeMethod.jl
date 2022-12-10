module FiniteVolumeMethod

using DelaunayTriangulation
using FunctionWrappersWrappers
using PreallocationTools
using StaticArraysCore
using LinearAlgebra
using SparseArrays
using SciMLBase
using DiffEqBase
import DiffEqBase: dualgen

include("geometry.jl")
include("boundary_conditions.jl")
include("problem.jl")
include("equations.jl")
include("solve.jl")
include("interpolant.jl")

export FVMGeometry
export BoundaryConditions
export FVMProblem
export eval_interpolant
export eval_interpolant!

end
