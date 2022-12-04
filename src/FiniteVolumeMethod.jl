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

include("fvm.jl")

export FVMGeometry
export BoundaryConditions
export FVMProblem
export FVMGeometry
export BoundaryConditions
export FVMProblem
export eval_interpolant
export eval_interpolant!

end
