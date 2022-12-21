# Docstrings

Here we give some of the main docstrings.

## FVMGeometry

```@docs 
FVMGeometry
FiniteVolumeMethod.MeshInformation
FiniteVolumeMethod.BoundaryInformation
FiniteVolumeMethod.BoundaryEdgeMatrix 
FiniteVolumeMethod.OutwardNormalBoundary
FiniteVolumeMethod.InteriorInformation 
FiniteVolumeMethod.construct_interior_edge_boundary_element_identifier
FiniteVolumeMethod.ElementInformation 
```

## BoundaryConditions 

```@docs 
BoundaryConditions 
FiniteVolumeMethod.is_dirichlet_type 
FiniteVolumeMethod.is_neumann_type
FiniteVolumeMethod.is_dudt_type
```

## FVMProblem 

```@docs 
FVMProblem 
FiniteVolumeMethod.construct_flux_function 
FiniteVolumeMethod.construct_reaction_function 
```

## Solving the FVMProblem 

```@docs 
FiniteVolumeMethod.solve 
FiniteVolumeMethod.ODEProblem 
FiniteVolumeMethod.jacobian_sparsity
```

## Linear Interpolants 

```@docs 
eval_interpolant 
```