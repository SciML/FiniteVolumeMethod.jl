using Documenter, FiniteVolumeMethod
DocMeta.setdocmeta!(FiniteVolumeMethod, :DocTestSetup, :(using FiniteVolumeMethod);
    recursive=true)

makedocs(;
    modules=[FiniteVolumeMethod],
    authors="Daniel VandenHeuvel <danj.vandenheuvel@gmail.com>",
    repo="https://github.com/DanielVandH/FiniteVolumeMethod.jl/blob/{commit}{path}#{line}",
    sitename="FiniteVolumeMethod.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://DanielVandH.github.io/FiniteVolumeMethod.jl",
        edit_link="main",
        assets=String[]),
    pages=[
        "Home" => "index.md",
        "Interface" => "interface.md",
        "Docstrings" => "docstrings.md",
        "Examples" => [
            "List of Examples and Setup" => "example_list.md",
            "Example I: Diffusion equation on a square plate" => "diffusion_equation.md",
            "Example II: Diffusion equation in a wedge with mixed boundary conditions" => "diffusion_equation_on_a_wedge.md",
            "Example III: Reaction-diffusion equation with a time-dependent Dirichlet boundary condition on a disk" => "reaction_diffusion.md",
            "Example IV: Porous-medium equation" => "porous_medium.md",
            "Example V: Porous-Fisher equation and travelling waves" => "porous_fisher.md",
            "Example VI: Using the linear interpolants" => "interpolants.md",
            "Example VII: Diffusion equation on an annulus" => "annulus.md",
            "Example VIII: Laplace's equation" => "laplace.md",
            "Example IX: Mean exit time problems" => "mean_exit_time.md",
        ],
        "Mathematical Details" => "math.md"
    ]
)

deploydocs(;
    repo="github.com/DanielVandH/FiniteVolumeMethod.jl",
    devbranch="main"
)