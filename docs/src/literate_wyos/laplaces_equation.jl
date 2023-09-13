# # Laplace's Equation 
# ```@contents 
# Pages = ["laplaces_equation.md"]
# ``` 
# Now we consider Laplace's equation. What we produce in this 
# section can also be accessed in `FiniteVolumeMethod.LaplacesEquation`.

# ## Mathematical Details 
# The mathematical details for this solver are the same as for 
# our [Poisson equation example](poissons_equation.md), except with 
# $f = 0$. The problems being solved are of the form 
# ```math 
# \div\left[D(\vb x)\grad u\right] = 0,
# ```
# known as the generalised Laplace equation.[^1]

# [^1]: See, for example, [this paper](https://doi.org/10.1016/0307-904X(87)90036-9) by Rangogni and Occhi (1987).

# ## Implementation 
# For the implementation, we can reuse a lot of what 
# we had for Poisson's equation.
using FiniteVolumeMethod, SparseArrays, DelaunayTriangulation, LinearSolve
const FVM = FiniteVolumeMethod
function laplaces_equation(mesh::FVMGeometry,
    BCs::BoundaryConditions,
    ICs::InternalConditions=InternalConditions();
    diffusion_function,
    diffusion_parameters=nothing)
    conditions = Conditions(mesh, BCs, ICs)
    n = DelaunayTriangulation.num_solid_vertices(mesh.triangulation)
    A = zeros(n, n)
    b = zeros(num_points(mesh.triangulation))
    FVM.triangle_contributions!(A, mesh, conditions, diffusion_function, diffusion_parameters)
    FVM.boundary_edge_contributions!(A, b, mesh, conditions, diffusion_function, diffusion_parameters)
   
end