module FiniteVolumeMethod

#=
include("fvm.jl")
=#
include("fast_fvm.jl")
export FVMGeometry
export BoundaryConditions
export FVMProblem

end
