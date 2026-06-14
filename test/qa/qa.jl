using FiniteVolumeMethod
using Test
using Aqua
using ExplicitImports

@testset verbose = true "Aqua" begin
    Aqua.test_all(FiniteVolumeMethod; ambiguities = false, project_extras = false) # don't care about julia < 1.2
    Aqua.test_ambiguities(FiniteVolumeMethod) # don't pick up Base and Core...
end

@testset verbose = true "Explicit Imports" begin
    @test check_no_implicit_imports(FiniteVolumeMethod) === nothing
    @test check_no_stale_explicit_imports(FiniteVolumeMethod) === nothing
end
