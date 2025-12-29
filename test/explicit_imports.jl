using ExplicitImports
using FiniteVolumeMethod
using Test

@test check_no_implicit_imports(FiniteVolumeMethod) === nothing
@test check_no_stale_explicit_imports(FiniteVolumeMethod) === nothing
