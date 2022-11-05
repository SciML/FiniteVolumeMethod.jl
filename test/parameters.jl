
for problem in (DiffusionEquationOnASquarePlate,
    ReactionDiffusiondudt,
    DiffusionOnAWedge,
    TravellingWaveProblem)
    for iip in [false, true]
        local prob, p
        prob, _ = problem(; iip_flux=iip)
        p = FVM.FVMParameters(prob)
        @test SciMLBase.isinplace(prob) == iip
        @test p.iip_flux == iip
        @test p.midpoints == prob.mesh.midpoints
        @test p.normals == prob.mesh.normals
        @test p.lengths == prob.mesh.lengths
        @test p.flux! == prob.flux!
        @test p.flux_vec.du == dualcache(zeros(2), 12).du
        @test p.flux_vec.dual_du == dualcache(zeros(2), 12).dual_du
        @test p.shape_coeffs.du == dualcache(zeros(3), 12).du
        @test p.shape_coeffs.dual_du == dualcache(zeros(3), 12).dual_du
        @test p.flux_params == prob.flux_parameters
        @test p.s == prob.mesh.shape_function_coeffs
        @test p.interior_or_neumann_nodes == prob.boundary_conditions.interior_or_neumann_nodes
        @test p.int_edge_bnd_el_id == prob.mesh.interior_edge_boundary_element_identifier
        @test p.react == prob.reaction
        @test p.react_params == prob.reaction_parameters
        @test p.points == prob.mesh.points
        @test p.volumes == prob.mesh.volumes
        @test p.dudt_tuples == prob.boundary_conditions.dudt_tuples
        @test p.bc_functions == prob.boundary_conditions.functions
        @test p.interior_elements == prob.mesh.interior_elements
        @test p.boundary_elements == prob.mesh.boundary_elements

        @test_throws "setfield!: const field" p.midpoints = Dict((1, 2, 3) => ([2.0, 2.0],[2.0,2.0],[2.0,2.0]))
        @test_throws "setfield!: const field" p.normals = Dict((1, 2, 3) => ([2.0, 2.0],[2.0,2.0],[2.0,2.0]))
        @test_throws "setfield!: const field" p.lengths = Dict((1, 2, 3) => (2, 3, 4))
        @test_throws "setfield!: const field" p.flux_vec = dualcache(zeros(2), 24)
        @test_throws "setfield!: const field" p.shape_coeffs = dualcache(zeros(2), 24)
        if problem == DiffusionEquationOnASquarePlate
            @test p.flux_params == 9
            p.flux_params = 271182
            @test p.flux_params ≠ 9
            @test p.flux_params == 271182
            @test_throws MethodError p.react_params = missing
        elseif problem == ReactionDiffusiondudt || problem == DiffusionOnAWedge
            @test_throws MethodError p.flux_params = missing
            @test_throws MethodError p.react_params = missing
        elseif problem == TravellingWaveProblem
            @test p.flux_params == 0.9
            p.flux_params = 1.9281
            @test p.flux_params == 1.9281
            @test p.react_params == 0.99
            p.react_params = 0.17771
            @test p.react_params == 0.17771
        end
        @test_throws "setfield!: const field" p.s = Dict((1, 2, 3) => (1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0))
        @test_throws "setfield!: const field" p.interior_or_neumann_nodes = [1, 2, 3]
        @test_throws "setfield!: const field" p.int_edge_bnd_el_id = p.int_edge_bnd_el_id
        @test_throws "setfield!: const field" p.points = p.points
        @test_throws "setfield!: const field" p.volumes = p.volumes
        @test_throws "setfield!: const field" p.dudt_tuples = p.dudt_tuples
        @test_throws "setfield!: const field" p.bc_functions = p.bc_functions
        @test_throws "setfield!: const field" p.interior_elements = p.interior_elements
        @test_throws "setfield!: const field" p.boundary_elements = p.boundary_elements
    end
end

prob, _ = DiffusionEquationOnASquarePlate()
p = FVM.FVMParameters(prob)
@test eltype(p.flux_vec.du) == Float64
@test eltype(p.shape_coeffs.du) == Float64
for T in [Float64, Int64, ForwardDiff.Dual]#[Float64, Num, Int64, ForwardDiff.Dual]
    local p
    p = FVM.FVMParameters(prob; cache_eltype=T)
    @test eltype(p.flux_vec.du) == T
    @test eltype(p.shape_coeffs.du) == T
end