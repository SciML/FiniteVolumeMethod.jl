module FiniteVolumeMethod

include("fvm.jl")

export FVMGeometry
export BoundaryConditions
export FVMProblem
export FVMParameters
export FVMInterpolant
export eval_interpolant
export points
export construct_mesh_interpolant
export construct_mesh_interpolant!

end
