@testset "Diffusion equation on a square plate" begin
    Random.seed!(299298823423423418)
    for iip = [false, true]
        prob, DTx, DTy = DiffusionEquationOnASquarePlate(; iip_flux=iip)

        ## Linear shape function coefficients 
        u = ones(length(prob.mesh.points))
        T = (305, 266, 437)
        αβγₖ = zeros(3)
        FVM.linear_shape_function_coefficients!(αβγₖ, u, prob.mesh.shape_function_coeffs, T)
        αₖ, βₖ, γₖ = αβγₖ
        @test αₖ + βₖ + γₖ ≈ 1.0
        u = rand(length(prob.mesh.points))
        FVM.linear_shape_function_coefficients!(αβγₖ, u, prob.mesh.shape_function_coeffs, T)
        αₖ, βₖ, γₖ = αβγₖ
        for i in 1:3
            @test αₖ * DTx[indices(T)[i]] + βₖ * DTy[indices(T)[i]] + γₖ ≈ u[indices(T)[i]]
        end
        αβγₖ = zeros(3)
        for T in prob.mesh.elements
            FVM.linear_shape_function_coefficients!(αβγₖ, u, prob.mesh.shape_function_coeffs, T)
            @test all(αβγₖ .== FVM.linear_shape_function_coefficients(u, prob.mesh.shape_function_coeffs, T))
        end

        ## Function over an edge
        Random.seed!(299291)
        u = rand(length(DTx))
        du = zeros(length(DTx))
        t = 0.0
        q = zeros(2)
        T = (313, 241, 450)
        j = getk(T)
        jnb = geti(T)
        FVM.linear_shape_function_coefficients!(αβγₖ, u, prob.mesh.shape_function_coeffs, T)
        αₖ, βₖ, γₖ = αβγₖ
        FVM.fvm_eqs_edge!(du, t, prob.mesh.midpoints, prob.mesh.normals, prob.mesh.lengths,
            prob.flux!, q, T, (j, 3), (jnb, 1), αₖ, βₖ, γₖ, prob.flux_parameters, prob.boundary_conditions.interior_or_neumann_nodes, iip)
        @test du[geti(T)] ≈ -0.0210371839790132082737006413708513719029724597930908203125
        @test du[getj(T)] == 0.0
        @test du[getk(T)] ≈ -du[geti(T)]

        ## Function over all edges
        Random.seed!(223291)
        u = rand(length(DTx))
        du = zeros(length(DTx))
        t = 0.0
        q = zeros(2)
        T = (296, 98, 419)
        FVM.linear_shape_function_coefficients!(αβγₖ, u, prob.mesh.shape_function_coeffs, T)
        αₖ, βₖ, γₖ = αβγₖ
        FVM.fvm_eqs_edge!(du, t, prob.mesh.midpoints, prob.mesh.normals, prob.mesh.lengths,
            prob.flux!, q, T, αₖ, βₖ, γₖ, prob.flux_parameters, prob.boundary_conditions.interior_or_neumann_nodes, iip)
        @test du[[T...]] ≈ [0.00494013657139753814551141886113327927887439727783203125,
            -0.0104251433246924894715501608288832358084619045257568359375,
            0.0054850067532949513260387419677499565295875072479248046875]

        ## Function over an interior element 
        Random.seed!(22374291)
        u = rand(length(DTx))
        du = zeros(length(DTx))
        t = 0.0
        q = zeros(2)
        T = (106, 281, 493)
        FVM.fvm_eqs_interior_element!(du, u, t, prob.mesh.midpoints, prob.mesh.normals, prob.mesh.lengths,
            prob.mesh.shape_function_coeffs, prob.flux!, q, T, αβγₖ, prob.flux_parameters, prob.boundary_conditions.interior_or_neumann_nodes, 0.0,
            iip)
        @test du[[T...]] ≈ [0.017665936857951487770623799633540329523384571075439453125,
            -0.021773099702587332682224285917982342652976512908935546875,
            0.00410716284463584491160048628444201312959194183349609375]

        ## Function over all interior elements 
        Random.seed!(22374291)
        u = rand(length(DTx))
        du = zeros(length(DTx))
        t = 0.0
        q = zeros(2)
        FVM.fvm_eqs_interior_element!(du, u, t, prob.mesh.midpoints, prob.mesh.normals, prob.mesh.lengths,
            prob.mesh.shape_function_coeffs, prob.flux!, q, αβγₖ, prob.flux_parameters, prob.boundary_conditions.interior_or_neumann_nodes,
            prob.mesh.interior_elements, iip)
        @test du[prob.boundary_conditions.dirichlet_nodes] == zeros(length(prob.boundary_conditions.dirichlet_nodes))
        @test du[[50, 101, 200, 201, 202, 493, 100, 500, 3, 5]] ≈ [
            0.0,
            -0.0905383028429150027438510051069897599518299102783203125,
            -0.12347092921786657904181794265241478569805622100830078125,
            0.008321215098178645630522254350580624304711818695068359375,
            0.1447608267817510341313180788347381167113780975341796875,
            0.012163791646424261527759114187574596144258975982666015625,
            -0.001171199456531983595919399476770195178687572479248046875,
            0.0282934387215102341184280732022671145386993885040283203125,
            0.0,
            0.0
        ]

        ## Identifying a boundary edge 
        T = (4, 5, 97) # see prob.mesh.boundary_elements
        res = prob.mesh.interior_edge_boundary_element_identifier[T]
        @test res == [((5, 2), (97, 3)), ((97, 3), (4, 1))]
        T = (78, 79, 109)
        res = prob.mesh.interior_edge_boundary_element_identifier[T]
        @test res == [((79, 2), (109, 3)), ((109, 3), (78, 1))]
        T = (40, 41, 513)
        res = prob.mesh.interior_edge_boundary_element_identifier[T]
        @test res == [((41, 2), (513, 3)), ((513, 3), (40, 1))]
        idx = convex_hull(prob.mesh.neighbours, prob.mesh.points)
        ch_edges = [[idx[i], idx[i == length(idx) ? 1 : i + 1]] for i in eachindex(idx)]
        for (T, res) in prob.mesh.interior_edge_boundary_element_identifier
            @test length(res) < 3
            for ((vj, j), (vjnb, jnb)) in res
                pj, pjnb = _get_point(prob.mesh.points, vj), _get_point(prob.mesh.points, vjnb)
                midpt = 0.5 * (pj + pjnb)
                @test [vj, vjnb] ∉ ch_edges
            end
        end

        ## Function over a boundary element 
        Random.seed!(2232345474291)
        u = rand(length(DTx))
        du = zeros(length(DTx))
        t = 0.5
        q = zeros(2)
        T = (80, 1, 498)
        FVM.fvm_eqs_boundary_element!(du, u, t, prob.mesh.midpoints, prob.mesh.normals, prob.mesh.lengths, prob.flux!,
            q, T, αβγₖ, prob.flux_parameters, prob.boundary_conditions.interior_or_neumann_nodes,
            prob.mesh.interior_edge_boundary_element_identifier, 0.0,
            prob.mesh.shape_function_coeffs, iip)
        @test du[[T...]] ≈ [0.0, 0.0, 0.028515199492761662825035529067463357932865619659423828125]
        T = (2, 3, 289)
        FVM.fvm_eqs_boundary_element!(du, u, t, prob.mesh.midpoints, prob.mesh.normals, prob.mesh.lengths, prob.flux!,
            q, T, αβγₖ, prob.flux_parameters, prob.boundary_conditions.interior_or_neumann_nodes,
            prob.mesh.interior_edge_boundary_element_identifier, 0.0,
            prob.mesh.shape_function_coeffs, iip)
        @test du[[T...]] ≈ [0.0, 0.0, -0.0175664590230240787172011351913170074112713336944580078125]
        T = (57, 58, 94)
        FVM.fvm_eqs_boundary_element!(du, u, t, prob.mesh.midpoints, prob.mesh.normals, prob.mesh.lengths, prob.flux!,
            q, T, αβγₖ, prob.flux_parameters, prob.boundary_conditions.interior_or_neumann_nodes,
            prob.mesh.interior_edge_boundary_element_identifier, 0.0,
            prob.mesh.shape_function_coeffs, iip)
        @test du[[T...]] ≈ [0.0, 0.0, 0.006160270904592065992211313840698494459502398967742919921875]

        ## Function over all boundary elements 
        Random.seed!(2232345474291)
        u = rand(length(DTx))
        du = zeros(length(DTx))
        t = 0.5
        q = zeros(2)
        FVM.fvm_eqs_boundary_element!(du, u, t, prob.mesh.midpoints, prob.mesh.normals, prob.mesh.lengths, prob.flux!,
            q, αβγₖ, prob.flux_parameters, prob.boundary_conditions.interior_or_neumann_nodes, prob.mesh.interior_edge_boundary_element_identifier,
            prob.mesh.boundary_elements, prob.mesh.shape_function_coeffs, iip)
        @test sum(du) ≈ -0.09870822201661655359572478118934668600559234619140625
        @test du[[1, 2, 4, 59, 101, 102, 98, 171, 231]] ≈ [
            0.0, 0.0, 0.0, 0.0,
            -0.0124361191715032133597862440410608542151749134063720703125,
            -0.015798488760815375397950077740460983477532863616943359375,
            -0.03856656690117567232167772317552589811384677886962890625,
            0.0, 0.0
        ]

        ## Function for the source contribution at a single term 
        Random.seed!(2232345474291)
        u = rand(length(DTx))
        du = rand(length(DTx))
        t = 0.5
        j = 37
        FVM.fvm_eqs_source_contribution!(du, u, t, j, prob.reaction, prob.reaction_parameters, prob.mesh.points, prob.mesh.volumes, 0.0)
        @test du[j] ≈ 68.8348555819915617348669911734759807586669921875
        j = 109
        FVM.fvm_eqs_source_contribution!(du, u, t, j, prob.reaction, prob.reaction_parameters, prob.mesh.points, prob.mesh.volumes, 0.0)
        @test du[j] ≈ 26.3954071751651468957788893021643161773681640625

        ## Function for all source contributions at all nodes
        Random.seed!(2232345474291)
        u = rand(length(DTx))
        du = rand(length(DTx))
        t = 0.5
        FVM.fvm_eqs_source_contribution!(du, u, t, prob.reaction, prob.reaction_parameters, prob.mesh.points, prob.mesh.volumes, prob.boundary_conditions.interior_or_neumann_nodes)
        @test du[[1, 2, 25, 101, 292, 81, 7, 111]] ≈ [
            0.32226187527768868079647290869615972042083740234375,
            0.7486476408295070772425106042646802961826324462890625,
            0.66843407762231354940496430572238750755786895751953125,
            47.27134400729307373012488824315369129180908203125,
            43.0237228577626211745155160315334796905517578125,
            44.69416128770516394297374063171446323394775390625,
            0.26263208841389940051413987021078355610370635986328125,
            68.991969409311394656469929032027721405029296875
        ]

        ## Test the FVM equations 
        Random.seed!(2232345474291)
        u = rand(length(DTx))
        du = rand(length(DTx))
        t = 0.5
        q = zeros(2)
        αβγₖ = zeros(3)
        p = FVM.FVMParameters(prob)
        FVM.fvm_eqs!(du, u, p, t)
        @test du[[1, 2, 9, 202, 101, 98, 57, 301, 50, 516]] ≈ [
            0.0,
            0.0,
            0.0,
            13.9839386663489211315436477889306843280792236328125,
            -4.56724711020632856417478251387365162372589111328125,
            -24.021591835212294796519927331246435642242431640625,
            0.0,
            -18.584341950546804156374491867609322071075439453125,
            0.0,
            -25.960313300633554689511584001593291759490966796875
        ]

        ## Updating Dirichlet boundary nodes 
        Random.seed!(2232345474291)
        u = ones(length(DTx))
        t = 0.3
        vals = prob.boundary_conditions.dirichlet_tuples
        F = prob.boundary_conditions.functions
        FVM.update_dirichlet_nodes!(u, t, vals, F)
        @test u[prob.boundary_conditions.dirichlet_nodes] == zeros(80)
    end
end

@testset "Diffusion on a wedge" begin
    Random.seed!(299298818)
    for iip = [true, false]
        prob, DTx, DTy, α = DiffusionOnAWedge(; iip_flux=iip)

        ## Linear shape function coefficients 
        u = ones(length(prob.mesh.points))
        T = (175, 164, 204)
        αβγₖ = zeros(3)
        FVM.linear_shape_function_coefficients!(αβγₖ, u, prob.mesh.shape_function_coeffs, T)
        αₖ, βₖ, γₖ = αβγₖ
        @test αₖ + βₖ + γₖ ≈ 1.0
        u = rand(length(prob.mesh.points))
        FVM.linear_shape_function_coefficients!(αβγₖ, u, prob.mesh.shape_function_coeffs, T)
        αₖ, βₖ, γₖ = αβγₖ
        for i in 1:3
            @test αₖ * DTx[indices(T)[i]] + βₖ * DTy[indices(T)[i]] + γₖ ≈ u[indices(T)[i]]
        end
        αβγₖ = zeros(3)
        for T in prob.mesh.elements
            FVM.linear_shape_function_coefficients!(αβγₖ, u, prob.mesh.shape_function_coeffs, T)
            @test all(αβγₖ .== FVM.linear_shape_function_coefficients(u, prob.mesh.shape_function_coeffs, T))
        end

        ## Function over an edge
        Random.seed!(299267391)
        u = rand(length(DTx))
        du = zeros(length(DTx))
        t = 0.0
        q = zeros(2)
        T = (159, 156, 218)
        j = 3
        jnb = 1
        FVM.linear_shape_function_coefficients!(αβγₖ, u, prob.mesh.shape_function_coeffs, T)
        αₖ, βₖ, γₖ = αβγₖ
        FVM.fvm_eqs_edge!(du, t, prob.mesh.midpoints, prob.mesh.normals, prob.mesh.lengths,
            prob.flux!, q, T, (218, j), (159, jnb), αₖ, βₖ, γₖ, prob.flux_parameters, prob.boundary_conditions.interior_or_neumann_nodes, iip)
        @test du[159] ≈ -0.005915182037392920956497821549646687344647943973541259765625
        @test du[156] == 0.0
        @test du[218] ≈ -du[159]

        ## Function over all edges
        Random.seed!(223266691)
        u = rand(length(DTx))
        du = zeros(length(DTx))
        t = 0.0
        q = zeros(2)
        T = (129, 190, 196)
        FVM.linear_shape_function_coefficients!(αβγₖ, u, prob.mesh.shape_function_coeffs, T)
        αₖ, βₖ, γₖ = αβγₖ
        FVM.fvm_eqs_edge!(du, t, prob.mesh.midpoints, prob.mesh.normals, prob.mesh.lengths,
            prob.flux!, q, T, αₖ, βₖ, γₖ, prob.flux_parameters, prob.boundary_conditions.interior_or_neumann_nodes, iip)
        @test du[[T...]] ≈ [
            0.3253412251457412640576194462482817471027374267578125,
            -0.03475947753509243953740082133663236163556575775146484375,
            -0.29058174761064881064243081709719263017177581787109375
        ]

        ## Function over an interior element 
        Random.seed!(22376454291)
        u = rand(length(DTx))
        du = zeros(length(DTx))
        t = 0.0
        q = zeros(2)
        T = (72, 53, 77)
        FVM.fvm_eqs_interior_element!(du, u, t, prob.mesh.midpoints, prob.mesh.normals, prob.mesh.lengths,
            prob.mesh.shape_function_coeffs, prob.flux!, q, T, αβγₖ, prob.flux_parameters, prob.boundary_conditions.interior_or_neumann_nodes, 0.0,
            iip)
        #=
        @test du[[T...]] ≈ [-0.04478116851101819617664290262837312184274196624755859375,
            0.31034892466514296671675765537656843662261962890625,
            -0.2655677561541247566623269449337385594844818115234375]
        =#
        ## Function over all interior elements 
        Random.seed!(22374653456291)
        u = rand(length(DTx))
        du = zeros(length(DTx))
        t = 0.0
        q = zeros(2)
        FVM.fvm_eqs_interior_element!(du, u, t, prob.mesh.midpoints, prob.mesh.normals, prob.mesh.lengths,
            prob.mesh.shape_function_coeffs, prob.flux!, q, αβγₖ, prob.flux_parameters, prob.boundary_conditions.interior_or_neumann_nodes,
            prob.mesh.interior_elements, iip)
        @test du[prob.boundary_conditions.dirichlet_nodes] == zeros(length(prob.boundary_conditions.dirichlet_nodes))
        @test du[[50, 101, 98, 171, 102, 1, 2, 3]] ≈ [
            0.34490351953365327997147460337146185338497161865234375,
            0.78045808027281859953205866986536420881748199462890625,
            1.3232666609761569720404850158956833183765411376953125,
            0.5872211885534011077680816015345044434070587158203125,
            1.477716219137004127759382754447869956493377685546875,
            0.0,
            0.0,
            0.0
        ]

        ## Identifying a boundary edge 
        T = (4, 5, 194)
        res = prob.mesh.interior_edge_boundary_element_identifier[T]
        @test res == [((5, 2), (194, 3)), ((194, 3), (4, 1))]
        T = (8, 9, 70)
        res = prob.mesh.interior_edge_boundary_element_identifier[T]
        @test res == [((9, 2), (70, 3)), ((70, 3), (8, 1))]
        idx = convex_hull(prob.mesh.neighbours, prob.mesh.points)
        ch_edges = [[idx[i], idx[i == length(idx) ? 1 : i + 1]] for i in eachindex(idx)]
        for (T, res) in prob.mesh.interior_edge_boundary_element_identifier
            @test length(res) < 3
            for ((vj, j), (vjnb, jnb)) in res
                pj, pjnb = _get_point(prob.mesh.points, vj), _get_point(prob.mesh.points, vjnb)
                midpt = 0.5 * (pj + pjnb)
                @test [vj, vjnb] ∉ ch_edges
            end
        end

        ## Function over a boundary element 
        Random.seed!(2232345474291)
        u = rand(length(DTx))
        du = zeros(length(DTx))
        t = 0.5
        q = zeros(2)
        T = (12, 13, 150)
        FVM.fvm_eqs_boundary_element!(du, u, t, prob.mesh.midpoints, prob.mesh.normals, prob.mesh.lengths, prob.flux!,
            q, T, αβγₖ, prob.flux_parameters, prob.boundary_conditions.interior_or_neumann_nodes, prob.mesh.interior_edge_boundary_element_identifier, 0.0,
            prob.mesh.shape_function_coeffs, iip)
        @test du[[T...]] ≈ [
            -0.0176313021366859869198950860891272895969450473785400390625
            0.047768186059016524980247453413539915345609188079833984375
            -0.0301368839223305380603523673244126257486641407012939453125
        ]
        T = (53, 54, 77)
        FVM.fvm_eqs_boundary_element!(du, u, t, prob.mesh.midpoints, prob.mesh.normals, prob.mesh.lengths, prob.flux!,
            q, T, αβγₖ, prob.flux_parameters, prob.boundary_conditions.interior_or_neumann_nodes, prob.mesh.interior_edge_boundary_element_identifier, 0.0,
            prob.mesh.shape_function_coeffs, iip)
        #=
        @test du[[T...]] ≈ [
            -0.0186789740260590413079011540276042069308459758758544921875
            0.09854367723115774191366966761052026413381099700927734375
            -0.079864703205098697136321561629301868379116058349609375
        ]
        =#
        #=
        T = (4, 56, 1)  
        #idx_order = [1, 2, 3]
        #T = T ∈ prob.mesh.elements ? T : DelaunayTriangulation.shift_triangle(T, 1) 
        #idx_order = [2, 3, 1]
        #T = T ∈ prob.mesh.elements ? T : DelaunayTriangulation.shift_triangle(T, 1)
        #idx_order = [3, 1, 2]
        FVM.fvm_eqs_boundary_element!(du, u, t, prob.mesh.midpoints, prob.mesh.normals, prob.mesh.lengths, prob.flux!,
            q, T, αβγₖ, prob.flux_parameters, prob.boundary_conditions.interior_or_neumann_nodes, prob.mesh.interior_edge_boundary_element_identifier, 0.0,
            prob.mesh.shape_function_coeffs, iip)
        @test du[[T...]] ≈ [-0.046001621750561015600222702914834371767938137054443359375
            0.046001621750561015600222702914834371767938137054443359375
            0.0]
        @test sum(du[[T...]]) ≈ 0.0
        =#

        ## Function over all boundary elements 
        Random.seed!(2232345474291)
        u = rand(length(DTx))
        du = zeros(length(DTx))
        t = 0.5
        q = zeros(2)
        FVM.fvm_eqs_boundary_element!(du, u, t, prob.mesh.midpoints, prob.mesh.normals, prob.mesh.lengths, prob.flux!,
            q, αβγₖ, prob.flux_parameters, prob.boundary_conditions.interior_or_neumann_nodes, prob.mesh.interior_edge_boundary_element_identifier,
            prob.mesh.boundary_elements, prob.mesh.shape_function_coeffs, iip)
        @test sum(du) ≈ -0.3578133991735656938004694893606938421726226806640625
        #=
        @test du[[1, 2, 4, 59, 101, 102, 98, 171, 200]] ≈ [
            0.0
            0.0
            -0.052322788902949841916001361141752568073570728302001953125
            0.309527065917022614627285292954184114933013916015625
            0.0
            0.0
            0.0
            0.079690389203384393024265364147140644490718841552734375
            -0.165690793748562936738011330817244015634059906005859375
        ]
        =#

        ## Function for the source contribution at a single term 
        Random.seed!(22323345474291)
        u = rand(length(DTx))
        du = rand(length(DTx))
        t = 0.5
        j = 37
        FVM.fvm_eqs_source_contribution!(du, u, t, j, prob.reaction, prob.reaction_parameters, prob.mesh.points, prob.mesh.volumes, 0.0)
        @test du[j] ≈ 695.5949282410483647254295647144317626953125
        j = 109
        FVM.fvm_eqs_source_contribution!(du, u, t, j, prob.reaction, prob.reaction_parameters, prob.mesh.points, prob.mesh.volumes, 0.0)
        @test du[j] ≈ 184.652618388309491592735867016017436981201171875

        ## Function for all source contributions at all nodes
        Random.seed!(224291)
        u = rand(length(DTx))
        du = rand(length(DTx))
        t = 0.5
        FVM.fvm_eqs_source_contribution!(du, u, t, prob.reaction, prob.reaction_parameters, prob.mesh.points, prob.mesh.volumes, prob.boundary_conditions.interior_or_neumann_nodes)
        @test du[[1, 2, 25, 101, 92, 81, 7, 111]] ≈ [
            467.85446382674052756556193344295024871826171875,
            0.72412048087447777877656562850461341440677642822265625,
            0.03670986217798688500124626443721354007720947265625,
            11.8515983817000769562355344532988965511322021484375,
            86.3020018137558651005747378803789615631103515625,
            194.94462625268585043158964253962039947509765625,
            867.90362994402175900177098810672760009765625,
            405.72163196430386733482009731233119964599609375
        ]

        ## Test the FVM equations 
        Random.seed!(2232345474291)
        u = rand(length(DTx))
        du = rand(length(DTx))
        t = 0.5
        p = FVM.FVMParameters(prob)
        FVM.fvm_eqs!(du, u, p, t)
        @test du[[1, 2, 9, 12, 101, 98, 57, 61, 50]] ≈ [
            0.0,
            0.0,
            -20.775753533891471391825689352117478847503662109375,
            -255.409496720566238536775927059352397918701171875,
            -285.43017072682158641327987425029277801513671875,
            -743.3267924661655570162110961973667144775390625,
            -532.8485789196998894112766720354557037353515625,
            -535.33198538751094019971787929534912109375,
            104.470065475967743395813158713281154632568359375
        ]

        ## Updating Dirichlet boundary nodes 
        Random.seed!(2232345474291)
        u = ones(length(DTx))
        t = 0.3
        vals = prob.boundary_conditions.dirichlet_tuples
        F = prob.boundary_conditions.functions
        FVM.update_dirichlet_nodes!(u, t, vals, F)
        @test u[prob.boundary_conditions.dirichlet_nodes] == zeros(17)
    end
end

@testset "Reaction-diffusion with du/dt condition" begin
    Random.seed!(2992983333818)
    for iip in [false, true]
        prob, DTx, DTy = ReactionDiffusiondudt(; iip_flux=iip)

        ## Linear shape function coefficients 
        u = ones(length(prob.mesh.points))
        T = (255, 81, 384)
        αβγₖ = zeros(3)
        FVM.linear_shape_function_coefficients!(αβγₖ, u, prob.mesh.shape_function_coeffs, T)
        αₖ, βₖ, γₖ = αβγₖ
        @test αₖ + βₖ + γₖ ≈ 1.0
        u = rand(length(prob.mesh.points))
        FVM.linear_shape_function_coefficients!(αβγₖ, u, prob.mesh.shape_function_coeffs, T)
        αₖ, βₖ, γₖ = αβγₖ
        for i in 1:3
            @test αₖ * DTx[indices(T)[i]] + βₖ * DTy[indices(T)[i]] + γₖ ≈ u[indices(T)[i]]
        end
        αβγₖ = zeros(3)
        for T in prob.mesh.elements
            FVM.linear_shape_function_coefficients!(αβγₖ, u, prob.mesh.shape_function_coeffs, T)
            @test all(αβγₖ .== FVM.linear_shape_function_coefficients(u, prob.mesh.shape_function_coeffs, T))
        end

        ## Function over an edge
        Random.seed!(299267391)
        u = rand(length(DTx))
        du = zeros(length(DTx))
        t = 0.0
        q = zeros(2)
        T = (173, 171, 174)
        j = 3
        jnb = 1
        FVM.linear_shape_function_coefficients!(αβγₖ, u, prob.mesh.shape_function_coeffs, T)
        αₖ, βₖ, γₖ = αβγₖ
        FVM.fvm_eqs_edge!(du, t, prob.mesh.midpoints, prob.mesh.normals, prob.mesh.lengths,
            prob.flux!, q, T, (174, j), (173, jnb), αₖ, βₖ, γₖ, prob.flux_parameters,
            prob.boundary_conditions.interior_or_neumann_nodes, iip)
        @test du[173] ≈ 0.0230020730632060944886863040892421850003302097320556640625
        @test du[171] == 0.0
        @test du[174] ≈ -du[173]

        ## Function over all edges
        Random.seed!(223266691)
        u = rand(length(DTx))
        du = zeros(length(DTx))
        t = 0.0
        q = zeros(2)
        T = (86, 95, 390)
        FVM.linear_shape_function_coefficients!(αβγₖ, u, prob.mesh.shape_function_coeffs, T)
        αₖ, βₖ, γₖ = αβγₖ
        FVM.fvm_eqs_edge!(du, t, prob.mesh.midpoints, prob.mesh.normals, prob.mesh.lengths,
            prob.flux!, q, T, αₖ, βₖ, γₖ, prob.flux_parameters, prob.boundary_conditions.interior_or_neumann_nodes, iip)
        @test du[[T...]] ≈ [
            -0.104952499280135835846294867224059998989105224609375,
            0.060730674150435960678695579417762928642332553863525390625,
            0.0442218251296998821064931917135254479944705963134765625
        ]

        ## Function over an interior element 
        Random.seed!(22376454291)
        u = rand(length(DTx))
        du = zeros(length(DTx))
        t = 0.0
        q = zeros(2)
        T = (357, 67, 363)
        FVM.fvm_eqs_interior_element!(du, u, t, prob.mesh.midpoints, prob.mesh.normals, prob.mesh.lengths,
            prob.mesh.shape_function_coeffs, prob.flux!, q, T, αβγₖ, prob.flux_parameters, prob.boundary_conditions.interior_or_neumann_nodes, 0.0,
            iip)
        @test du[[T...]] ≈ [-0.00522115722743443817777109217104225535877048969268798828125,
            0.024001930682574841580123603534957510419189929962158203125,
            -0.0187807734551404016676290353871081606484949588775634765625]

        ## Function over all interior elements 
        Random.seed!(22374653456291)
        u = rand(length(DTx))
        du = zeros(length(DTx))
        t = 0.0
        q = zeros(2)
        FVM.fvm_eqs_interior_element!(du, u, t, prob.mesh.midpoints, prob.mesh.normals, prob.mesh.lengths,
            prob.mesh.shape_function_coeffs, prob.flux!, q, αβγₖ, prob.flux_parameters, prob.boundary_conditions.interior_or_neumann_nodes,
            prob.mesh.interior_elements,
            iip)
        @test du[prob.boundary_conditions.dirichlet_nodes] == zeros(length(prob.boundary_conditions.dirichlet_nodes))
        @test du[[50, 101, 98, 171, 102, 1, 2, 3]] ≈ [
            0.0,
            0.18692826092352843314614574410370551049709320068359375,
            0.5251208191272811465211134418495930731296539306640625,
            0.7441750432861089503688845070428214967250823974609375,
            0.286492218096898010326611938580754213035106658935546875,
            0.0,
            0.0,
            0.0
        ]

        ## Identifying a boundary edge 
        T = (31, 32, 69)
        res = prob.mesh.interior_edge_boundary_element_identifier[T]
        @test res == [((32, 2), (69, 3)), ((69, 3), (31, 1))]
        T = (8, 9, 381)
        res = prob.mesh.interior_edge_boundary_element_identifier[T]
        @test res == [((9, 2), (381, 3)), ((381, 3), (8, 1))]
        idx = convex_hull(prob.mesh.neighbours, prob.mesh.points)
        ch_edges = [[idx[i], idx[i == length(idx) ? 1 : i + 1]] for i in eachindex(idx)]
        for (T, res) in prob.mesh.interior_edge_boundary_element_identifier
            @test length(res) < 3
            for ((vj, j), (vjnb, jnb)) in res
                pj, pjnb = _get_point(prob.mesh.points, vj), _get_point(prob.mesh.points, vjnb)
                midpt = 0.5 * (pj + pjnb)
                @test [vj, vjnb] ∉ ch_edges
            end
        end

        ## Function over a boundary element 
        Random.seed!(223691)
        u = rand(length(DTx))
        du = zeros(length(DTx))
        t = 0.5
        q = zeros(2)
        T = (59, 60, 402)
        FVM.fvm_eqs_boundary_element!(du, u, t, prob.mesh.midpoints, prob.mesh.normals, prob.mesh.lengths, prob.flux!,
            q, T, αβγₖ, prob.flux_parameters, prob.boundary_conditions.interior_or_neumann_nodes, prob.mesh.interior_edge_boundary_element_identifier, 0.0,
            prob.mesh.shape_function_coeffs, iip)
        @test du[[T...]] ≈ [
            0.0,
            0.0,
            0.1489716616993781
        ]
        T = (2, 3, 95)
        FVM.fvm_eqs_boundary_element!(du, u, t, prob.mesh.midpoints, prob.mesh.normals, prob.mesh.lengths, prob.flux!,
            q, T, αβγₖ, prob.flux_parameters, prob.boundary_conditions.interior_or_neumann_nodes, prob.mesh.interior_edge_boundary_element_identifier, 0.0,
            prob.mesh.shape_function_coeffs, iip)
        @test du[[T...]] ≈ [
            0.0
            0.0
            -0.1902065942651491037285182983396225608885288238525390625
        ]
        T = (7, 8, 65)
        FVM.fvm_eqs_boundary_element!(du, u, t, prob.mesh.midpoints, prob.mesh.normals, prob.mesh.lengths, prob.flux!,
            q, T, αβγₖ, prob.flux_parameters, prob.boundary_conditions.interior_or_neumann_nodes, prob.mesh.interior_edge_boundary_element_identifier, 0.0,
            prob.mesh.shape_function_coeffs, iip)
        @test du[[T...]] ≈ [
            0.0
            0.0
            0.0257616448184504291674112863574919174425303936004638671875
        ]

        ## Function over all boundary elements 
        Random.seed!(2232345474291)
        u = rand(length(DTx))
        du = zeros(length(DTx))
        t = 0.5
        q = zeros(2)
        FVM.fvm_eqs_boundary_element!(du, u, t, prob.mesh.midpoints, prob.mesh.normals, prob.mesh.lengths, prob.flux!,
            q, αβγₖ, prob.flux_parameters, prob.boundary_conditions.interior_or_neumann_nodes, prob.mesh.interior_edge_boundary_element_identifier,
            prob.mesh.boundary_elements,
            prob.mesh.shape_function_coeffs, iip)
        @test sum(du) ≈ -0.364809758491894287057988321976154111325740814208984375
        @test du[[64, 65, 66, 402, 403, 1, 2, 3]] ≈ [
            -0.1984176933139911447323555648836190812289714813232421875
            0.0222563398007851291626746359497701632790267467498779296875
            0.08426048744693755143675417684789863415062427520751953125
            -0.136808408028720795979182867085910402238368988037109375
            -0.268545649612721260002246026488137431442737579345703125
            0.0
            0.0
            0.0
        ]

        ## Function for the source contribution at a single term 
        Random.seed!(22323345474291)
        u = rand(length(DTx))
        du = rand(length(DTx))
        t = 0.5
        j = 37
        FVM.fvm_eqs_source_contribution!(du, u, t, j, prob.reaction, prob.reaction_parameters, prob.mesh.points, prob.mesh.volumes, 0.0)
        @test du[j] ≈ 231.6898182490764384056092239916324615478515625
        j = 109
        FVM.fvm_eqs_source_contribution!(du, u, t, j, prob.reaction, prob.reaction_parameters, prob.mesh.points, prob.mesh.volumes, 0.0)
        @test du[j] ≈ 6.243804658957689213139019557274878025054931640625

        ## Function for all source contributions at all nodes
        Random.seed!(224291)
        u = rand(length(DTx))
        du = rand(length(DTx))
        t = 0.5
        FVM.fvm_eqs_source_contribution!(du, u, t, prob.reaction, prob.reaction_parameters, prob.mesh.points, prob.mesh.volumes, prob.boundary_conditions.interior_or_neumann_nodes)
        @test du[[1, 2, 25, 101, 92, 81, 7, 111]] ≈ [
            0.71674912327502127151745980881969444453716278076171875
            0.919949100938604491517480710172094404697418212890625
            0.90951297733832436875900384620763361454010009765625
            60.862272739926339681915123946964740753173828125
            93.1625257124144781073482590727508068084716796875
            68.264195704403078934774384833872318267822265625
            0.68274508170726588840437898397794924676418304443359375
            21.9897508102984744482455425895750522613525390625
        ]

        ## Evaluating the dudt boundary conditions 
        Random.seed!(224291)
        u = rand(length(DTx))
        du = zeros(length(DTx))
        t = 0.5
        vals = prob.boundary_conditions.dudt_tuples
        F = prob.boundary_conditions.functions
        FVM.update_dudt_nodes!(du, u, t, vals, F)
        @test du[prob.boundary_conditions.dudt_nodes] ≈ u[prob.boundary_conditions.dudt_nodes]

        ## Test the FVM equations 
        Random.seed!(2232345474291)
        u = rand(length(DTx))
        du = rand(length(DTx))
        t = 0.5
        p = FVM.FVMParameters(prob)
        FVM.fvm_eqs!(du, u, p, t)
        @test du[[1, 2, 9, 12, 101, 98, 57, 61, 50]] ≈ [
            0.839687250525305461934522099909372627735137939453125,
            0.7461450744459863226865081742289476096630096435546875,
            0.09925239210209280127372721835854463279247283935546875,
            0.990202912158163695011126037570647895336151123046875,
            -31.8721114308656154889831668697297573089599609375,
            -128.1868472429008534163585864007472991943359375,
            0.64575189463319848659494937237468548119068145751953125,
            0.9077873215588183608559802451054565608501434326171875,
            0.5599246715853942735208192971185781061649322509765625
        ]

        ## Updating Dirichlet boundary nodes 
        Random.seed!(2232345474291)
        u = ones(length(DTx))
        t = 0.3
        vals = prob.boundary_conditions.dirichlet_tuples
        F = prob.boundary_conditions.functions
        FVM.update_dirichlet_nodes!(u, t, vals, F)
        @test u[prob.boundary_conditions.dirichlet_nodes] == Float64[]
    end
end
