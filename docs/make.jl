using Distributed # https://github.com/CliMA/Oceananigans.jl/blob/main/docs/make.jl
Distributed.addprocs(2)

@everywhere begin
    pushfirst!(LOAD_PATH, joinpath(@__DIR__, ".."))
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    Pkg.instantiate()
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
end

@everywhere begin
    using FiniteVolumeMethod
    using Documenter
    using Literate
    using Dates
    ct() = Dates.format(now(), "HH:MM:SS")
    using CairoMakie
    CairoMakie.activate!()

    # When running docs locally, the EditURL is incorrect. For example, we might get 
    #   ```@meta
    #   EditURL = "<unknown>/docs/src/literate_tutorials/name.jl"
    #   ```
    # We need to replace this EditURL if we are running the docs locally. The last case is more complicated because, 
    # after changing to use temporary directories, it can now look like...
    #   ```@meta
    #   EditURL = "../../../../../../../AppData/Local/Temp/jl_8nsMGu/name_just_the_code.jl"
    #   ```
    function update_edit_url(content, file, folder)
        content = replace(content, "<unknown>" => "https://github.com/DanielVandH/FiniteVolumeMethod.jl/tree/main")
        content = replace(content, "temp/" => "") # as of Literate 2.14.1
        content = replace(content, r"EditURL\s*=\s*\"[^\"]*\"" => "EditURL = \"https://github.com/DanielVandH/FiniteVolumeMethod.jl/tree/main/docs/src/literate_$(folder)/$file\"")
        return content
    end
    # We can add the code to the end of each file in its uncommented form programatically.
    function add_just_the_code_section(dir, file)
        file_name, file_ext = splitext(file)
        file_path = joinpath(dir, file)
        new_file_path = joinpath(session_tmp, file_name * "_just_the_code" * file_ext)
        cp(file_path, new_file_path, force=true)
        folder = splitpath(dir)[end] # literate_tutorials or literate_applications
        open(new_file_path, "a") do io
            write(io, "\n")
            write(io, "# ## Just the code\n")
            write(io, "# An uncommented version of this example is given below.\n")
            write(io, "# You can view the source code for this file [here](<unknown>/docs/src/$folder/@__NAME__.jl).\n")
            write(io, "\n")
            write(io, "# ```julia\n")
            write(io, "# @__CODE__\n")
            write(io, "# ```\n")
        end
        return new_file_path
    end

    tutorial_files = [
        #"tutorials/gray_scott_model_turing_patterns_from_a_coupled_reaction_diffusion_system.jl",
        #"tutorials/mean_exit_time.jl",
        #"tutorials/solving_mazes_with_laplaces_equation.jl",
        #"tutorials/porous_medium_equation.jl",
        #"tutorials/equilibrium_temperature_distribution_with_mixed_boundary_conditions_and_using_ensembleproblems.jl",
        #"tutorials/reaction_diffusion_brusselator_system_of_pdes.jl",
        #"tutorials/diffusion_equation_on_a_square_plate.jl",
        #"tutorials/diffusion_equation_in_a_wedge_with_mixed_boundary_conditions.jl",
        #"tutorials/reaction_diffusion_equation_with_a_time_dependent_dirichlet_boundary_condition_on_a_disk.jl",
        #"tutorials/porous_fisher_equation_and_travelling_waves.jl",
        #"tutorials/piecewise_linear_and_natural_neighbour_interpolation_for_an_advection_diffusion_equation.jl",
        #"tutorials/helmholtz_equation_with_inhomogeneous_boundary_conditions.jl",
        #"tutorials/laplaces_equation_with_internal_dirichlet_conditions.jl",
        #"tutorials/diffusion_equation_on_an_annulus.jl",
    ]
    wyos_files = [
        #"wyos/diffusion_equations.jl",
        #"wyos/laplaces_equation.jl",
        #"wyos/mean_exit_time.jl",
        "wyos/poissons_equation.jl",
        #"wyos/linear_reaction_diffusion_equations.jl"
    ]
    example_files = vcat(tutorial_files, wyos_files)
    session_tmp = mktempdir()
end

Distributed.pmap(1:length(example_files)) do n
    example = example_files[n]
    folder, file = splitpath(example)
    dir = joinpath(@__DIR__, "src", "literate_" * folder)
    outputdir = joinpath(@__DIR__, "src", folder)
    file_path = joinpath(dir, file)
    # See also https://github.com/Ferrite-FEM/Ferrite.jl/blob/d474caf357c696cdb80d7c5e1edcbc7b4c91af6b/docs/generate.jl for some of this
    new_file_path = add_just_the_code_section(dir, file)
    script = Literate.script(file_path, session_tmp, name=splitext(file)[1] * "_just_the_code_cleaned")
    code = strip(read(script, String))
    @info "[$(ct())] Processing $file: Converting markdown script"
    line_ending_symbol = occursin(code, "\r\n") ? "\r\n" : "\n"
    code_clean = join(filter(x -> !endswith(x, "#hide"), split(code, r"\n|\r\n")), line_ending_symbol)
    code_clean = replace(code_clean, r"^# This file was generated .*$"m => "")
    code_clean = strip(code_clean)
    post_strip = content -> replace(content, "@__CODE__" => code_clean)
    editurl_update = content -> update_edit_url(content, file, folder)
    IS_LIVESERVER = false # get(ENV, "LIVESERVER_ACTIVE", "false") == "true"
    Literate.markdown(
        new_file_path,
        outputdir;
        documenter=true,
        postprocess=editurl_update ∘ post_strip,
        credit=true,
        execute=!IS_LIVESERVER,
        flavor=Literate.DocumenterFlavor(),
        name=splitext(file)[1]
    )
end

Distributed.rmprocs()

# All the pages to be included
_PAGES = [
    "Introduction" => "index.md",
    "Interface" => "interface.md",
    "Tutorials" => [
        "Section Overview" => "tutorials/overview.md",
        #"Diffusion Equation on a Square Plate" => "tutorials/diffusion_equation_on_a_square_plate.md",
        #"Diffusion Equation in a Wedge with Mixed Boundary Conditions" => "tutorials/diffusion_equation_in_a_wedge_with_mixed_boundary_conditions.md",
        #"Reaction-Diffusion Equation with a Time-dependent Dirichlet Boundary Condition on a Disk" => "tutorials/reaction_diffusion_equation_with_a_time_dependent_dirichlet_boundary_condition_on_a_disk.md",
        #"Porous-Medium Equation" => "tutorials/porous_medium_equation.md",
        #"Porous-Fisher Equation and Travelling Waves" => "tutorials/porous_fisher_equation_and_travelling_waves.md",
        #"Piecewise Linear and Natural Neighbour Interpolation for an Advection-Diffusion Equation" => "tutorials/piecewise_linear_and_natural_neighbour_interpolation_for_an_advection_diffusion_equation.md",
        #"Helmholtz Equation with Inhomogeneous Boundary Conditions" => "tutorials/helmholtz_equation_with_inhomogeneous_boundary_conditions.md",
        #"Laplace's Equation with Internal Dirichlet Conditions" => "tutorials/laplaces_equation_with_internal_dirichlet_conditions.md",
        #"Equilibrium Temperature Distribution with Mixed Boundary Conditions and using EnsembleProblems" => "tutorials/equilibrium_temperature_distribution_with_mixed_boundary_conditions_and_using_ensembleproblems.md",
        #"A Reaction-Diffusion Brusselator System of PDEs" => "tutorials/reaction_diffusion_brusselator_system_of_pdes.md",
        #"Gray-Scott Model: Turing Patterns from a Coupled Reaction-Diffusion System" => "tutorials/gray_scott_model_turing_patterns_from_a_coupled_reaction_diffusion_system.md",
        #"Diffusion Equation on an Annulus" => "tutorials/diffusion_equation_on_an_annulus.md",
        #"Mean Exit Time" => "tutorials/mean_exit_time.md",
        #"Solving Mazes with Laplace's Equation" => "tutorials/solving_mazes_with_laplaces_equation.md",
        "Keller-Segel Model of Chemotaxis" => "tutorials/keller_segel_chemotaxis.md",
    ],
    "Solvers for Specific Problems, and Writing Your Own" => [
        "Section Overview" => "wyos/overview.md",
        #"Diffusion Equations" => "wyos/diffusion_equations.md",
        #"Mean Exit Time Problems" => "wyos/mean_exit_time.md",
        #"Linear Reaction-Diffusion Equations" => "wyos/linear_reaction_diffusion_equations.md",
        "Poisson's Equation" => "wyos/poissons_equation.md",
        #"Laplace's Equation" => "wyos/laplaces_equation.md",
    ],
    "Mathematical and Implementation Details" => "math.md"
]

# Make sure we haven't forgotten any files
set = Set{String}()
for page in _PAGES
    if page[2] isa String
        push!(set, normpath(page[2]))
    else
        for _page in page[2]
            if _page[2] isa String
                push!(set, normpath(_page[2]))
            else
                for __page in _page[2]
                    push!(set, normpath(__page[2]))
                end
            end
        end
    end
end
missing_set = String[]
doc_dir = joinpath(@__DIR__, "src", "")
for (root, dir, files) in walkdir(doc_dir)
    for file in files
        filename = normpath(replace(joinpath(root, file), doc_dir => ""))
        if endswith(filename, ".md") && filename ∉ set
            push!(missing_set, filename)
        end
    end
end
!isempty(missing_set) && error("Missing files: $missing_set")

# Make and deploy
DocMeta.setdocmeta!(FiniteVolumeMethod, :DocTestSetup, :(using FiniteVolumeMethod, Test);
    recursive=true)
IS_LIVESERVER = false # get(ENV, "LIVESERVER_ACTIVE", "false") == "true"
IS_CI = get(ENV, "CI", "false") == "true"
makedocs(;
    modules=[FiniteVolumeMethod],
    authors="Daniel VandenHeuvel <danj.vandenheuvel@gmail.com>",
    sitename="FiniteVolumeMethod.jl",
    format=Documenter.HTML(;
        prettyurls=IS_CI,
        canonical="https://DanielVandH.github.io/FiniteVolumeMethod.jl",
        edit_link="main",
        collapselevel=1,
        assets=String[],
        mathengine=MathJax3(Dict(
            :loader => Dict("load" => ["[tex]/physics"]),
            :tex => Dict(
                "inlineMath" => [["\$", "\$"], ["\\(", "\\)"]],
                "tags" => "ams",
                "packages" => ["base", "ams", "autoload", "physics"],
            ),
        ))),
    draft=IS_LIVESERVER,
    pages=_PAGES,
    warnonly=true
)

deploydocs(;
    repo="github.com/DanielVandH/FiniteVolumeMethod.jl",
    devbranch="main",
    push_preview=true)

function clear_tmp()
    rm(session_tmp, force=true, recursive=true)
    protected_tutorials = ["overview.md", "keller_segel_chemotaxis.md", "maze.txt"]
    protected_wyos = ["overview.md"]
    protected = Dict("tutorials" => protected_tutorials, "wyos" => protected_wyos)
    for folder in ("tutorials", "wyos")
        dir = joinpath(@__DIR__, "src", folder)
        files = readdir(dir)
        setdiff!(files, protected[folder])
        for file in files
            rm(joinpath(dir, file))
        end
    end
end
clear_tmp()
