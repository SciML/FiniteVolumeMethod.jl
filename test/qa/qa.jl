using SciMLTesting, FiniteVolumeMethod, Test

run_qa(
    FiniteVolumeMethod;
    explicit_imports = true,
    # Root Project.toml carries only `Test` in [extras]/[targets]; the real test
    # deps live in test/Project.toml under the grouped-tests folder model, so the
    # root-vs-test consistency check does not apply here.
    aqua_kwargs = (; project_extras = false),
)
