using SciMLTesting, FiniteVolumeMethod, Test

@test SciMLTesting.public_reexports(FiniteVolumeMethod) == [:solve]

run_qa(
    FiniteVolumeMethod;
    explicit_imports = true,
    # `solve` is reexported from CommonSolve; this package documents its
    # methods but does not own the generic binding.
    reexports_allow = (:solve,),
    api_docs_kwargs = (; rendered = true, rendered_ignore = (:solve,)),
    # Root Project.toml carries only `Test` in [extras]/[targets]; the real test
    # deps live in test/Project.toml under the grouped-tests folder model, so the
    # root-vs-test consistency check does not apply here.
    aqua_kwargs = (; project_extras = false),
)
