```@meta
EditURL = "https://github.com/DanielVandH/FiniteVolumeMethod.jl/tree/main/docs/src/literate_tutorials/solving_mazes_with_laplaces_equation.jl"
```

# Solving Mazes with Laplace's Equation
In this [tutorial](solving_mazes_with_laplaces_equation.md), we consider solving
mazes using Laplace's equation, applying the result of
[Conolly, Burns, and Weis (1990)](https://doi.org/10.1109/ROBOT.1990.126315).
In particular, given a maze $\mathcal M$, represented as a collection of edges together with some starting point
$\mathcal S_1$ and an endpoint $\mathcal S_2$,
Laplace's equation can be used to find the solution:
```math
\begin{equation}
\begin{aligned}
\grad^2 \phi &= 0, & \vb x \in \mathcal M, \\
\phi &= 0 & \vb x \in \mathcal S_1, \\
\phi &= 1 & \vb x \in \mathcal S_2, \\
\grad\phi\vdot\vu n &= 0 & \vb x \in \partial M \setminus (\mathcal S_1 \cup \mathcal S_2).
\end{aligned}
\end{equation}
```
The gradient $\grad\phi$ will reveal the solution to the maze.

For the first maze,

````@example solving_mazes_with_laplaces_equation
using DelaunayTriangulation, CairoMakie, DelimitedFiles
readdlm("maze.txt")
````

## Just the code
An uncommented version of this example is given below.
You can view the source code for this file [here](https://github.com/DanielVandH/FiniteVolumeMethod.jl/tree/new-docs/docs/src/literate_tutorials/solving_mazes_with_laplaces_equation.jl).

```julia
using DelaunayTriangulation, CairoMakie, DelimitedFiles
readdlm("maze.txt")
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

