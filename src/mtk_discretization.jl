"""
    FVMDiscretization

A discretization type for the finite volume method on unstructured triangular meshes.

This type is used with ModelingToolkit's `discretize` interface to convert symbolic
PDESystem specifications into FVMProblem instances. The actual discretization logic
is implemented in the FiniteVolumeMethodModelingToolkitExt extension, which is loaded
when both FiniteVolumeMethod and ModelingToolkit are available.

# Fields

  - `mesh::FVMGeometry`: The finite volume mesh geometry
  - `boundary_condition_map`: Optional mapping of PDESystem boundary regions to mesh boundary indices

# Constructor

    FVMDiscretization(mesh::FVMGeometry; boundary_condition_map=nothing)

# Usage

```julia
using FiniteVolumeMethod, ModelingToolkit, DelaunayTriangulation

# Create mesh
tri = triangulate_rectangle(0, 1, 0, 1, 20, 20, single_boundary = true)
mesh = FVMGeometry(tri)

# Create discretization
disc = FVMDiscretization(mesh)

# Use with PDESystem
prob = discretize(pdesys, disc)
```

# Notes

Since FiniteVolumeMethod uses unstructured triangular meshes from DelaunayTriangulation.jl,
the mesh must be provided separately from the PDESystem domain specification. The PDESystem
domains are used for extracting time spans, but the spatial discretization comes from the mesh.

For boundary conditions with multiple boundary segments, you may need to provide
a `boundary_condition_map` to specify which PDESystem boundary conditions apply to which
mesh boundary segments.

# Supported PDE Forms

When used with ModelingToolkit, the following PDE forms are supported:

  - Diffusion equations: ∂u/∂t = ∇·(D∇u)
  - Reaction-diffusion equations: ∂u/∂t = ∇·(D∇u) + R(u)

For more general PDEs, use FVMProblem directly with custom flux and source functions.

See also [`FVMProblem`](@ref), [`FVMGeometry`](@ref), [`BoundaryConditions`](@ref).
"""
struct FVMDiscretization{M <: FVMGeometry, B} <: SciMLBase.AbstractDiscretization
    mesh::M
    boundary_condition_map::B
end

function FVMDiscretization(mesh::FVMGeometry; boundary_condition_map = nothing)
    return FVMDiscretization(mesh, boundary_condition_map)
end

function Base.show(io::IO, ::MIME"text/plain", disc::FVMDiscretization)
    nv = DelaunayTriangulation.num_solid_vertices(disc.mesh.triangulation)
    print(io, "FVMDiscretization with mesh of $(nv) nodes")
end
