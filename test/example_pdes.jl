@testset "Diffusion equation on a square plate" begin
    for iip in [false, true]
        prob, DTx, DTy = DiffusionEquationOnASquarePlate(; iip_flux=iip)
        ## Solve the problem 
        sol = solve(prob, prob.solver)
        u_fvm = reduce(hcat, sol.u)
        t = sol.t
        ## Compare solutions 
        time = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        time_idx = Vector{Int64}([])
        for i in eachindex(time)
            idx = findmin(abs.(t .- time[i]))[2]
            push!(time_idx, idx)
        end
        ## Now do some actual testing 
        errs = Vector{Array{Float64}}([])
        errs_vec = Vector{Float64}([])
        all_abs_errs = Vector{Float64}([])
        for i in eachindex(time)
            soln_approx = sol.u[time_idx[i]]
            soln_exact = DiffusionEquationOnASquarePlateExact(DTx, DTy, time[i], 200, 200)
            err = error_measure(soln_exact, soln_approx)
            push!(errs, err)
            push!(errs_vec, err...)
            push!(all_abs_errs, abs.(soln_approx - soln_exact)...)
        end
        @test [mean(e) for e in errs] ≈ [0.0
            0.9944976650871220069660694207414053380489349365234375
            1.40171952328728988135253530344925820827484130859375
            1.3003460296715585453597441301099024713039398193359375
            1.0033750702266530652195797301828861236572265625
            0.903203087338031007647032311069779098033905029296875] rtol = 1e-5
        @test mean(all_abs_errs) ≈ 0.430375593567100922509638394330977462232112884521484375 rtol = 1e-5
        @test median(all_abs_errs) ≈ 0.104346566097678561391148832626640796661376953125 rtol = 1e-5
        algs = [AutoTsit5 ∘ Rosenbrock23,
            AutoVern7 ∘ Rodas4,
            AutoVern7 ∘ KenCarp4,
            AutoVern7 ∘ Rodas5,
            AutoVern9 ∘ Rodas4,
            AutoVern9 ∘ KenCarp4,
            AutoVern9 ∘ Rodas5,
            Tsit5,
            BS5,
            OwrenZen5,
            BS3,
            OwrenZen3,
            Vern6,
            Vern7,
            Vern8,
            Vern9,
            VCABM,
            RK4,
            Rosenbrock23,
            TRBDF2,
            #QNDF,
            #FBDF,
            Rodas5,
            Rodas4P,
            Kvaerno5,
            KenCarp4][[1, 2, 3, 4, 7, 15, 19]]
        errs = [
            0.0093333419405715982442028888499407912604510784149169921875
            0.00933353058430441695492163489689119160175323486328125
            0.00933353058430441695492163489689119160175323486328125
            0.00933353058430441695492163489689119160175323486328125
            0.009333530811240324520650091244533541612327098846435546875
            0.009333530811240324520650091244533541612327098846435546875
            0.009333530811240324520650091244533541612327098846435546875
            0.0093333419405715982442028888499407912604510784149169921875
            0.009333564692488159619809806599732837639749050140380859375
            0.00933351612238490259410017557684113853611052036285400390625
            0.0093214476032150439532841801337781362235546112060546875
            0.009333447991622424722013562359279603697359561920166015625
            0.009333541232358515127298659308507922105491161346435546875
            0.00933353058430441695492163489689119160175323486328125
            0.00933352757001056597407906423313761479221284389495849609375
            0.009333530811240324520650091244533541612327098846435546875
            0.0094002689848009350626067970324584166519343852996826171875
            0.0093343678863142438839606285228001070208847522735595703125
            0.00938053746760011443461824143241756246425211429595947265625
            0.01014369962369159224035985999989861738868057727813720703125
            0.0092810485648996653151865388053920469246804714202880859375
            0.0093334794278998843564121301596969715319573879241943359375
            0.0093338161872583598477337574195189517922699451446533203125
            0.009298992506812708835894909498165361583232879638671875
            0.0092321805814939879308855807948930305428802967071533203125
        ][[1, 2, 3, 4, 7, 15, 19]]
        time = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]
        for (i, alg) in pairs(algs)
            sol = solve(prob, alg())
            err = compute_errors(sol, t -> DiffusionEquationOnASquarePlateExact(DTx, DTy, t), time)
            @test err ≈ errs[i] atol = 1e-5
        end
    end
end

@testset "Diffusion on a wedge" begin
    for iip in [false, true]
        prob, DTx, DTy, α = DiffusionOnAWedge(; iip_flux=iip)
        # Solve the problem 
        sol = solve(prob)
        u_fvm = reduce(hcat, sol.u)
        t = sol.t
        ## Times for comparing the solutions
        times = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]
        time_idx = Vector{Int64}([])
        for i in eachindex(times)
            idx = findmin(abs.(t .- times[i]))[2]
            push!(time_idx, idx)
        end
        ## Now do some actual testing 
        errs = Vector{Array{Float64}}([])
        errs_vec = Vector{Float64}([])
        all_abs_errs = Vector{Float64}([])
        for i in eachindex(times)
            soln_approx = sol.u[time_idx[i]]
            soln_exact = DiffusionOnAWedgeExact(DTx, DTy, times[i], α, 22, 24)
            err = error_measure(soln_exact, soln_approx)
            push!(errs, err)
            push!(errs_vec, err...)
            push!(all_abs_errs, abs.(soln_approx - soln_exact)...)
        end
        @test [mean(e) for e in errs] ≈ [0.03905144109747747671601558749898686073720455169677734375
            1.459456275188696139366584247909486293792724609375
            1.8962984120646784180763688709703274071216583251953125
            2.28947336448180838175403550849296152591705322265625
            4.0120550223142412704646631027571856975555419921875
            1.685695494202481992118691778159700334072113037109375] rtol = 1e-3
        @test mean(all_abs_errs) ≈ 0.0106766130777785316074979249378884560428559780120849609375 rtol = 1e-3
        @test median(all_abs_errs) ≈ 0.00485664461568117988843340526727843098342418670654296875 rtol = 1e-3
        ## Test many algorithms 
        algs = [AutoTsit5 ∘ Rosenbrock23,
            AutoVern7 ∘ Rodas4,
            AutoVern7 ∘ KenCarp4,
            AutoVern7 ∘ Rodas5,
            AutoVern9 ∘ Rodas4,
            AutoVern9 ∘ KenCarp4,
            AutoVern9 ∘ Rodas5,
            Tsit5,
            BS5,
            OwrenZen5,
            BS3,
            OwrenZen3,
            Vern6,
            Vern7,
            Vern8,
            Vern9,
            VCABM,
            RK4,
            Rosenbrock23,
            TRBDF2,
            #QNDF,
            #FBDF,
            Rodas5,
            Rodas4P,
            Kvaerno5,
            KenCarp4][[1, 2, 3, 4, 7, 15, 19]]
        errs = [
            0.0873629091256813150589977112758788280189037322998046875
            0.08739657207175233200047159698442555963993072509765625
            0.0870033504977380245382079237970174290239810943603515625
            0.087396651288634996657123110708198510110378265380859375
            0.08739720108701830303399304966660565696656703948974609375
            0.0870278859256170267411079066732781939208507537841796875
            0.0873971775455835153678663118625991046428680419921875
            0.087477326001087429840907816469552926719188690185546875
            0.089223510765069236061464152953703887760639190673828125
            0.0873971754919284082863129015095182694494724273681640625
            0.08784070101555309373342339540613465942442417144775390625
            0.0872765066599397176805297249302384443581104278564453125
            0.0870666338628076352801343773535336367785930633544921875
            0.087421373231614418752855044658645056188106536865234375
            0.08779152182506484713986338874747161753475666046142578125
            0.0873971538208665743585612517563276924192905426025390625
            0.0881969425351556146619458331770147196948528289794921875
            0.0873752221762314207342825511659611947834491729736328125
            0.0870933993505126624601331286612548865377902984619140625
            0.0921345476620000491951856247396790422499179840087890625
            0.0915355187139088588565982718137092888355255126953125
            0.08756719769267252984068505838877172209322452545166015625
            0.08738109090307609461145688101169071160256862640380859375
            0.08733553673155014518414418489555828273296356201171875
            0.0869798964859874457200561437275609932839870452880859375
        ][[1, 2, 3, 4, 7, 15, 19]]
        times = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]
        for (i, alg) in pairs(algs)
            sol = solve(prob, alg())
            err = compute_errors(sol, t -> DiffusionOnAWedgeExact(DTx, DTy, t, α), times)
            @test err ≈ errs[i] atol = 1e-3
        end
    end
end

@testset "Reaction-diffusion with du/dt condition" begin
    for iip in [false, true]
        prob, DTx, DTy = ReactionDiffusiondudt(; iip_flux=iip)
        # Solve the problem 
        sol = solve(prob; saveat=0.01)
        u_fvm = reduce(hcat, sol.u)
        t = sol.t
        ## Times for comparing the solutions
        time = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]
        time_idx = Vector{Int64}([])
        for i in eachindex(time)
            idx = findmin(abs.(t .- time[i]))[2]
            push!(time_idx, idx)
        end
        ## Now do some tests
        errs = Vector{Array{Float64}}([])
        errs_vec = Vector{Float64}([])
        all_abs_errs = Vector{Float64}([])
        for i in eachindex(time)
            soln_approx = sol.u[time_idx[i]]
            soln_exact = ReactionDiffusiondudtExact(DTx, DTy, time[i])
            err = error_measure(soln_exact, soln_approx)
            push!(errs, err)
            push!(errs_vec, err...)
            push!(all_abs_errs, abs.(soln_approx - soln_exact)...)
        end
        @test [mean(e) for e in errs] ≈ [0.0
            0.004720814983895032547478454176825835020281374454498291015625
            0.00607752421518026157698511013904862920753657817840576171875
            0.00978387538164124208328598086836791480891406536102294921875
            0.00979498798973023852842967329479506588540971279144287109375
            0.007475913818963997141409105751108654658310115337371826171875] rtol = 1e-3
        @test mean(all_abs_errs) ≈ 8.427030552104572305553709110625959510798566043376922607421875e-05 rtol = 1e-3
        @test median(all_abs_errs) ≈ 5.352153060311781729296853882260620594024658203125e-05 rtol = 1e-3
        ## Test many algorithms 
        algs = [AutoTsit5 ∘ Rosenbrock23,
            AutoVern7 ∘ Rodas4,
            AutoVern7 ∘ KenCarp4,
            AutoVern7 ∘ Rodas5,
            AutoVern9 ∘ Rodas4,
            AutoVern9 ∘ KenCarp4,
            AutoVern9 ∘ Rodas5,
            Tsit5,
            BS5,
            OwrenZen5,
            BS3,
            OwrenZen3,
            Vern6,
            Vern7,
            Vern8,
            Vern9,
            VCABM,
            RK4,
            Rosenbrock23,
            TRBDF2,
            #QNDF,
            #FBDF,
            Rodas5,
            Rodas4P,
            Kvaerno5,
            KenCarp4][[1, 2, 3, 4, 7, 15, 19]]
        errs = [
            0.00407653565255731466232536064353553229011595249176025390625
            0.004007814930026818746322536668458269559778273105621337890625
            0.00400485271600984314710469647025092854164540767669677734375
            0.004007814930026818746322536668458269559778273105621337890625
            4.288605370030165175876391003839671611785888671875
            4.288605370030165175876391003839671611785888671875
            4.288605370030165175876391003839671611785888671875
            0.00406131912409513073924927084590308368206024169921875
            0.00418569931089870887752635297829328919760882854461669921875
            0.00402241047952182038949242581793441786430776119232177734375
            0.00407640237669272542664344882723526097834110260009765625
            0.00409342402468854289765420872981849242933094501495361328125
            0.0040761226876022856047132592038906295783817768096923828125
            4.288605370030165175876391003839671611785888671875
            4.288605370030165175876391003839671611785888671875
            4.288605370030165175876391003839671611785888671875
            0.005478267268622507256414788656684322631917893886566162109375
            0.00394118334700030263639813910003795172087848186492919921875
            0.00404132355139728838278045941478922031819820404052734375
            0.00427486794795411788328021174265813897363841533660888671875
            0.018992544553648287031233365951266023330390453338623046875
            0.0040526649165168772415146491994164534844458103179931640625
            0.0040635416014417058416796635356149636209011077880859375
            0.00395110629442123155452559757350172731094062328338623046875
            0.00395702467766312349084500965545885264873504638671875
        ][[1, 2, 3, 4, 7, 15, 19]]
        times = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]
        for (i, alg) in pairs(algs)
            sol = solve(prob, alg())
            err = compute_errors(sol, t -> ReactionDiffusiondudtExact(DTx, DTy, t), times)
            @test err ≈ errs[i] atol = 1e-3
        end
    end
end

@testset "Travelling wave problem" begin
    for iip in [false, true]
        prob, DTx, DTy, diffus, prolif, Nx, Ny, a, b, c, d = TravellingWaveProblem(; iip_flux=iip)
        C, D = c, d
        ## Solve the problem 
        #LinearAlgebra.BLAS.set_num_threads(1)
        sol = solve(prob)
        u_fvm = reduce(hcat, sol.u)
        t = sol.t
        ## Times for computing the solutions
        times = LinRange(0, prob.final_time, 6)
        time_idx = Vector{Int64}([])
        for i in eachindex(times)
            idx = findmin(abs.(t .- times[i]))[2]
            push!(time_idx, idx)
        end
        ## Now do some tests
        x_vals = @. a + ((1:Nx) - 1) * (b - a) / (Nx - 1)
        y_vals = @. c + ((1:Ny) - 1) * (d - c) / (Ny - 1)
        # Solution values
        @test sol.u[time_idx[1]] == zeros(600)
        @test [sum(sol.u[time_idx[i]]) for i in 1:6] ≈ [
            0.0
            118.447584793466461405841982923448085784912109375
            226.80922510097667554873623885214328765869140625
            335.87365611568048961999011225998401641845703125
            442.8160325876004890233161859214305877685546875
            553.31083008983705440186895430088043212890625
        ] rtol = 1e-5
        # x-invariance 
        function x_comparisons(i)
            solns = reshape(sol.u[time_idx[i]], (Nx, Ny))
            errs = Vector{Float64}([])
            for k in 1:Nx
                for j in 1:Nx
                    push!(errs, sum(abs.(solns[k, :] - solns[j, :])))
                end
            end
            return median(errs)
        end
        true_vals = [0.0
            0.027801205194556470601252584629037301056087017059326171875
            0.029490465304181055772314579144222079776227474212646484375
            0.03010694822566102601957283013689448125660419464111328125
            0.031911461980863094212157449192091007716953754425048828125
            0.02832787634045620672740284362589591182768344879150390625]
        for i in 1:6
            @test abs(x_comparisons(i) - true_vals[i]) < 1e-3
        end
        # Compare to exact solution 
        time_idx_10 = findfirst(sol.t .≥ 10.0)
        times = sol.t[time_idx_10:length(sol.t)]
        col = 8
        c = sqrt(prolif / (2diffus))
        cₘᵢₙ = sqrt(prolif * diffus / 2)
        zᶜ = 1.6
        exact_soln4 = z -> z ≤ zᶜ ? 1 - exp(cₘᵢₙ * (z - zᶜ)) : 0.0
        for i in time_idx_10:length(sol.t)
            solns = reshape(sol.u[i], (Nx, Ny))[col, :]
            z = @. y_vals - c * sol.t[i]
            exact = exact_soln4.(z)
            @test median(abs.(exact - solns)) < 1e-3
        end
        ## More algorithms 
        algs = [AutoTsit5 ∘ Rosenbrock23,
            AutoVern7 ∘ Rodas4,
            AutoVern7 ∘ KenCarp4,
            AutoVern7 ∘ Rodas5,
            AutoVern9 ∘ Rodas4,
            AutoVern9 ∘ KenCarp4,
            AutoVern9 ∘ Rodas5,
            Rodas5]
        times = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
        errs = [0.02025204660208533569143440900006680749356746673583984375
            0.02041780546960658693222967485780827701091766357421875
            0.020425881323910710651858835262828506529331207275390625
            0.0204141967927387313341824892631848342716693878173828125
            0.0204202839981514859546418705349424271844327449798583984375
            0.0204314063703251898307389211595364031381905078887939453125
            0.0204146138246752394118654905241783126257359981536865234375
            0.02041740491870609741642539347594720311462879180908203125]
        errs2 = [0.00107081361095262561633489895029924809932708740234375
            0.0013437594614182923891121390624903142452239990234375
            0.0015775442878706302796132376897730864584445953369140625
            0.003050970495819760319733404685393907129764556884765625
            0.00091586243030017389088470736169256269931793212890625
            0.00105218118307981445269660980557091534137725830078125
            0.002400014088148327839888906964915804564952850341796875
            0.0013246541650043519577906181439175270497798919677734375]
        for (i, alg) in pairs(algs)
            sol = solve(prob, alg())
            err = TravellingWaveProblemInvarianceErrors(sol, times, Nx, Ny)
            err2 = median(TravellingWaveLongTimeErrors(sol, 10.0, 8, 1.6, prolif, diffus, C, D, Nx, Ny))
            @test err ≈ errs[i] atol = 1e-3
            @test err2 ≈ errs2[i] atol = 1e-3
        end
    end
end


times = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]
fig = Figure(fontsize=31, resolution=(1450, 1200))
plot_idx = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
tris = collect(prob.mesh.elements)
tris = [tris[i][j] for i in eachindex(tris), j in 1:3]
ax = Axis(fig[1, 1])
for i in eachindex(times)
    ax = Axis(fig[plot_idx[i][1], plot_idx[i][2]], xlabel=L"x",
        ylabel=L"y", title=L"t = %$(times[i])", titlealign=:left,
        width=350, height=350, aspect=1)
    mesh!(ax, nodes, tris, color=errs[i], colormap=:viridis, colorrange=(0, 1))
end
Colorbar(fig[0, 1:3], limits=(0, 1), colormap=:viridis, labelsize=54, label=L"u(x, y)", vertical=false)