using Documenter, FiniteVolumeMethod 

makedocs(sitename = "FiniteVolumeMethod.jl",
modules = [FiniteVolumeMethod],
pages = [
    "Home" => "index.md"
    "Interface" => "interface.md"
    "Docstrings" => "docstrings.md"
    "List of Examples and Setup" => "example_list.md"
    "Example I: Diffusion equation on a square plate" => "diffusion_equation.md"
    "Example II: Diffusion equation in a wedge with mixed boundary conditions" => "diffusion_equation_on_a_wedge.md"
    "Example III: Reaction-diffusion equation with a time-dependent Dirichlet boundary condition on a disk" => "reaction_diffusion.md"
    "Example IV: Porous-medium equation" => "porous_medium.md"
    "Example V: Porous-Fisher equation and travelling waves" => "porous_fisher.md"
    "Example VI: Using the linear interpolants" => "interpolants.md"
    "Mathematical Details" => "math.md"
])

deploydocs(;
    repo="github.com/DanielVandH/FiniteVolumeMethod.jl",
    devbranch="main"
)