using FiniteVolumeMethod
using Documenter
using Literate
using Dates
_PAGES = [
    "Introduction" => "index.md",
    "Interface" => "interface.md",
    "Tutorials" => [
        "Section Overview" => "tutorials/overview.md",
        "Diffusion Equation on a Square Plate" => "tutorials/diffusion_equation_on_a_square_plate.md",
        "Diffusion Equation in a Wedge with Mixed Boundary Conditions" => "tutorials/diffusion_equation_in_a_wedge_with_mixed_boundary_conditions.md",
        "Reaction-Diffusion Equation with a Time-dependent Dirichlet Boundary Condition on a Disk" => "tutorials/reaction_diffusion_equation_with_a_time_dependent_dirichlet_boundary_condition_on_a_disk.md",
        "Porous-Medium Equation" => "tutorials/porous_medium_equation.md",
        "Porous-Fisher Equation and Travelling Waves" => "tutorials/porous_fisher_equation_and_travelling_waves.md",
        "Piecewise Linear and Natural Neighbour Interpolation for an Advection-Diffusion Equation" => "tutorials/piecewise_linear_and_natural_neighbour_interpolation_for_an_advection_diffusion_equation.md",
        "Helmholtz Equation with Inhomogeneous Boundary Conditions" => "tutorials/helmholtz_equation_with_inhomogeneous_boundary_conditions.md",
        "Laplace's Equation with Internal Dirichlet Conditions" => "tutorials/laplaces_equation_with_internal_dirichlet_conditions.md",
        "Equilibrium Temperature Distribution with Mixed Boundary Conditions and using EnsembleProblems" => "tutorials/equilibrium_temperature_distribution_with_mixed_boundary_conditions_and_using_ensembleproblems.md",
        "A Reaction-Diffusion Brusselator System of PDEs" => "tutorials/reaction_diffusion_brusselator_system_of_pdes.md",
        "Gray-Scott Model: Turing Patterns from a Coupled Reaction-Diffusion System" => "tutorials/gray_scott_model_turing_patterns_from_a_coupled_reaction_diffusion_system.md",
        "Diffusion Equation on an Annulus" => "tutorials/diffusion_equation_on_an_annulus.md",
        "Mean Exit Time" => "tutorials/mean_exit_time.md",
        "Solving Mazes with Laplace's Equation" => "tutorials/solving_mazes_with_laplaces_equation.md",
        "Keller-Segel Model of Chemotaxis" => "tutorials/keller_segel_chemotaxis.md",
    ],
    "Solvers for Specific Problems, and Writing Your Own" => [
        "Section Overview" => "wyos/overview.md",
        "Diffusion Equations" => "wyos/diffusion_equations.md",
        "Mean Exit Time Problems" => "wyos/mean_exit_time.md",
        "Linear Reaction-Diffusion Equations" => "wyos/linear_reaction_diffusion_equations.md",
        "Poisson's Equation" => "wyos/poissons_equation.md",
        "Laplace's Equation" => "wyos/laplaces_equation.md",
    ],
    "Mathematical and Implementation Details" => "math.md"
]
DocMeta.setdocmeta!(FiniteVolumeMethod, :DocTestSetup, :(using FiniteVolumeMethod, Test);
    recursive=true)
IS_LIVESERVER = get(ENV, "LIVESERVER_ACTIVE", "false") == "true"
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