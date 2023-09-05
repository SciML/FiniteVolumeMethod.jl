# FiniteVolumeMethod

[![DOI](https://zenodo.org/badge/561533716.svg)](https://zenodo.org/badge/latestdoi/561533716)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://DanielVandH.github.io/FiniteVolumeMethod.jl/dev)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://DanielVandH.github.io/FiniteVolumeMethod.jl/stable)
[![Coverage](https://codecov.io/gh/DanielVandH/FiniteVolumeMethod.jl/branch/main/graph/badge.svg?token=XPM5KN89R6)](https://codecov.io/gh/DanielVandH/FiniteVolumeMethod.jl)

This is a Julia package for solving partial differential equations (PDEs) of the form 

$$
\dfrac{\partial u(\boldsymbol x, t)}{\partial t} + \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{q}(\boldsymbol x, t, u) = S(\boldsymbol x, t, u), \quad (x, y)^{\mkern-1.5mu\mathsf{T}} \in \Omega \subset \mathbb R^2,t>0,
$$

in two dimensions using the finite volume method, with support also provided for steady-state problems and for systems of PDEs of the above form. 

If this package doesn't suit what you need, you may like to review some of the other PDE packages shown [here](https://github.com/JuliaPDE/SurveyofPDEPackages).

Please see the docs for more information.
