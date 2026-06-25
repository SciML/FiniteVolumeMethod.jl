using SciMLTesting, FiniteVolumeMethod, Test

run_qa(
    FiniteVolumeMethod;
    explicit_imports = true,
    # Root Project.toml carries only `Test` in [extras]/[targets]; the real test
    # deps live in test/Project.toml under the grouped-tests folder model, so the
    # root-vs-test consistency check does not apply here.
    aqua_kwargs = (; project_extras = false),
    ei_kwargs = (;
        # Other packages' non-public names accessed via `Module.name`. Each goes
        # public as its base lib releases; ignore until then.
        all_qualified_accesses_are_public = (;
            ignore = (
                :AutoSpecialize,                      # SciMLBase
                :each_point, :each_point_index,       # DelaunayTriangulation
                :get_centroid, :get_edge_midpoints,   # DelaunayTriangulation
                :has_boundary_nodes, :has_ghost_triangles,  # DelaunayTriangulation
                :has_vertex, :is_ghost_vertex,        # DelaunayTriangulation
                :num_points, :num_segments,           # DelaunayTriangulation
                :front, :tail,                        # Base
                :init, :solve,                        # CommonSolve
            ),
        ),
        # `solve` is CommonSolve's canonical (non-public) extension point that
        # FiniteVolumeMethod imports and re-exports.
        all_explicit_imports_are_public = (; ignore = (:solve,)),
    ),
)
