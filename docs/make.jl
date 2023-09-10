using FiniteVolumeMethod
using Documenter
using Literate
using Dates

DocMeta.setdocmeta!(FiniteVolumeMethod, :DocTestSetup, :(using FiniteVolumeMethod, Test);
    recursive=true)

const IS_LIVESERVER = get(ENV, "LIVESERVER_ACTIVE", "false") == "true"
if IS_LIVESERVER
    using Revise
    Revise.revise()
end
const IS_GITHUB_ACTIONS = get(ENV, "GITHUB_ACTIONS", "false") == "true"
const IS_CI = get(ENV, "CI", "false") == "true"
const session_tmp = mktempdir()

# When running docs locally, the EditURL is incorrect. For example, we might get 
#   ```@meta
#   EditURL = "<unknown>/docs/src/literate_tutorials/constrained.jl"
#   ```
# We need to replace this EditURL if we are running the docs locally. The last case is more complicated because, 
# after changing to use temporary directories, it can now look like...
#   ```@meta
#   EditURL = "../../../../../../../AppData/Local/Temp/jl_8nsMGu/cs1_just_the_code.jl"
#   ```
function update_edit_url(content, file, folder)
    content = replace(content, "<unknown>" => "https://github.com/DanielVandH/FiniteVolumeMethod.jl/tree/new-docs")
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

# Now process all the literate files
ct() = Dates.format(now(), "HH:MM:SS")
folder = "tutorials"
dir = joinpath(@__DIR__, "src", "literate_" * folder)
outputdir = joinpath(@__DIR__, "src", folder)
!isdir(outputdir) && mkpath(outputdir)
files = readdir(dir)
filter!(file -> endswith(file, ".jl") && !occursin("just_the_code", file), files)
for file in files
    # See also https://github.com/Ferrite-FEM/Ferrite.jl/blob/d474caf357c696cdb80d7c5e1edcbc7b4c91af6b/docs/generate.jl for some of this
    file_path = joinpath(dir, file)
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
    Literate.markdown(
        new_file_path,
        outputdir;
        documenter=true,
        postprocess=editurl_update ∘ post_strip,
        credit=true,
        name=splitext(file)[1]
    )
end

# All the pages to be included
const _PAGES = [
    "Introduction" => "index.md",
    "Interface" => "interface.md",
    "Tutorials" => [
        #"Section Overview" => "tutorials/overview.md",
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
        "Solving Mazes with Laplace's Equation" => "tutorials/solving_mazes_with_laplaces_equation.md",
        #"Keller-Segel Model of Chemotaxis" => "tutorials/keller_segel_chemotaxis.md"
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
# !isempty(missing_set) && error("Missing files: $missing_set")

# Make and deploy
makedocs(;
    modules=[FiniteVolumeMethod],
    authors="Daniel VandenHeuvel <danj.vandenheuvel@gmail.com>",
    repo="https://github.com/DanielVandH/FiniteVolumeMethod.jl/blob/{commit}{path}#{line}",
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
    pages=_PAGES
)

deploydocs(;
    repo="github.com/DanielVandH/FiniteVolumeMethod.jl",
    devbranch="main")