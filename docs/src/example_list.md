# List of Examples and Setup

We provide six examples in the sidebar. For running these examples, it is assumed that you have installed Gmsh, and have it available. I currently have set

```julia
GMSH_PATH = "./gmsh-4.9.4-Windows64/gmsh.exe"
```

The following packages are also loaded:

```julia
using FiniteVolumeMethod
using DelaunayTriangulation
using OrdinaryDiffEq 
using LinearSolve 
using CairoMakie 
using Bessels
using Test
using StatsBase
using SteadyStateDiffEq 
using LinearAlgebra
using NonlinearSolve 
using FastLapackInterface
using Krylov
```

More detail is given in the tests. In particular, if you want to see *just* the code, then see the corresponding test files.

We list all the examples below, but solve them in their respective sections of the sidebar.

## Example I: Diffusion equation on a square plate 

This example concerns the solution of the diffusion equation on a square plate, 

```math 
\begin{equation*}
\begin{array}{rcll}
\displaystyle
\frac{\partial u(x, y, t)}{\partial t} &=& \dfrac19\boldsymbol{\nabla}^2 u(x, y, t) & (x, y) \in \Omega,t>0, \\
u(x, y, t) &= & 0 & (x, y) \in \partial \Omega,t>0, \\
u(x, y, 0) &= & f(x, y) &(x,y)\in\Omega,
\end{array}
\end{equation*}
```

where $\Omega = [0, 2]^2$ and $f(x, y) = 50$ if $y \leq 1$ and $f(x, y) = 0$ if $y>1$.

## Example II: Diffusion equation in a wedge with mixed boundary conditions 

This example considers the following problem defined on a wedge with angle $\alpha$ and mixed boundary conditions:

```math
\begin{equation*}
\begin{array}{rcll}
\dfrac{\partial u(r, \theta, t)}{\partial t} & = & \boldsymbol{\nabla}^2 u(r, \theta, t), & 0 < r < 1, 0 < \theta < \alpha, t>0, \\
\dfrac{\partial u(r, 0, t)}{\partial \theta} & = & 0, & 0 < r < 1, t>0, \\
\dfrac{\partial u(r, \alpha,t)}{\partial \theta} & = & 0, & 0 < \theta < \alpha, t>0, \\
u(r,\theta,0) & = & f(r, \theta), & 0 < r < 1, 0< \theta < \alpha.
\end{array}
\end{equation*}
```

We let $f(r, \theta) = 1-r$.

## Example III: Reaction-diffusion equation with a time-dependent Dirichlet boundary condition on a disk 

This example considers the following reaction-diffusion equation on a disk:

```math 
\begin{equation*}
\begin{array}{rcll}
\dfrac{\partial u(r, \theta, t)}{\partial t} & = & \boldsymbol{\nabla} \boldsymbol{\cdot} [u\boldsymbol{\nabla} u] + u(1-u), & 0 < r < 1, 0 < \theta < 2\mathrm{\pi}, \\
\dfrac{\mathrm{d}u(1, \theta, t)}{\mathrm{d}t} & = & u(1, \theta, t), & 0 < \theta < 2\mathrm{\pi}, t > 0,  \\
u(r, \theta, 0) & = & \sqrt{I_0(\sqrt{2}r)},
\end{array}
\end{equation*}
```

where $I_0$ is the modified Bessel function of the first kind of order zero.

## Example IV: Porous-medium equation 

In this example, the Porous-Medium equation 

```math 
\dfrac{\partial u}{\partial t} = D\boldsymbol{\nabla} \boldsymbol{\cdot} \left[u^{m-1} \boldsymbol{\nabla}u\right],
```

is considered, where $u(x, y, 0) = M\delta(x, y)$, $\delta(x, y)$ is the Dirac delta function, and $M = \iint_{\mathbb R^2} u(x, y, t)~\mathrm{d}A$. We then extend this to consider 

```math 
\dfrac{\partial u}{\partial t} = D\boldsymbol{\nabla} \boldsymbol{\cdot} \left[u^{m-1}\boldsymbol{\nabla} u\right] + \lambda u, \quad \lambda>0. 
```

## Example V: Porous-Fisher equation and travelling waves 

In this example, the Porous-Fisher equation on a tall rectangle is considered:

```math 
\begin{equation*}
\begin{array}{rcll}
\dfrac{\partial u(x, y, t)}{\partial t} &=& D\boldsymbol{\nabla} \boldsymbol{\cdot} [u \boldsymbol{\nabla} u] + \lambda u(1-u), & 0 < x < a, 0 < y < b, t > 0, \\
u(x, 0, t) & = & 1 & 0 < x < a, t > 0, \\
u(x, b, t) & = & 0 & 0 < x < a , t > 0, \\
\dfrac{\partial u(0, y, t)}{\partial x} & = & 0 & 0 < y < b, t > 0, \\
\dfrac{\partial u(a, y, t)}{\partial x} & = & 0 & 0 < y < b, t > 0, \\
u(x, y, 0) & = & f(y) & 0 \leq x \leq a, 0 \leq y \leq b. 
\end{array}
\end{equation*}
```

We treat this problem numerically and consider it to a travelling wave from the analogous one-dimensional problem.

## Example VI: Using the linear interpolants 

The purpose of this example is to demonstrate how to efficiently make use of the linear interpolant defined by the method, and we demonstrate it on the same PDE as in Example V.

## Example VII: Diffusion equation on an annulus 

The purpose of this example is to show how we can solve problems on a multiply-connected domain, namely an annulus. The problem we solve is 

```math
\begin{equation*}
\begin{array}{rcll}
\dfrac{\partial u}{\partial t} & = & \boldsymbol{\nabla}^2 u & \boldsymbol{x} \in \Omega, t>0, \\
\boldsymbol{\nabla} u \boldsymbol{\cdot}  \hat{\boldsymbol n} = 0 & \boldsymbol{x} \in \mathcal D(0, 1), t>0, \\
u(x, y, t) & = & c(t) & \boldsymbol{x} \in \mathcal D(0, 0.2),  t>0, \\
u(x, y, 0) & = & u_0(x, y) & \boldsymbol{x} \in \Omega,
\end{array}
\end{equation*}
```

where $\mathcal D(0, r)$ is a circle of radius $r$ centred at the origin, $\Omega$ is the annulus bteween $\mathcal D(0, 0.2)$ and $\mathcal D(0, 1)$, $c(t) = 50[1-\exp(-0.5t)]$, and 

```math 
u_0(x) =10\mathrm{e}^{-25\left(\left(x + \frac12\right)^2 + \left(y + \frac12\right)^2\right)} - 10\mathrm{e}^{-45\left(\left(x - \frac12\right)^2 + \left(y - \frac12\right)^2\right)} - 5\mathrm{e}^{-50\left(\left(x + \frac{3}{10}\right)^2 + \left(y + \frac12\right)^2\right)}.
```

## Example VIII: Laplace's equation

The purpose of this example is to show how we can solve a steady problem. We solve Laplace's equation, given by 

```math
\begin{equation*}
\begin{array}{rcll}
\boldsymbol{\nabla}^2 u(x, y) & = & 0 & 0 < x, y < \pi, \\
u(x, 0) & = & \sinh(x) & 0 < x < \pi,
u(x, \pi) & = & -\sinh(x) & 0 < x < \pi, 
u(0, y) & = & 0 & 0 < y < \pi, \\
u(\pi, y) & = & \sinh(\pi)\cos(y) & 0 < x < \pi,
\end{array}
\end{equation*}
```

The exact solution to this problem is $u(x, y) = \sinh(x)\cos(y)$.

## Example IX: Mean exit time problems 

Here we give a further demonstration of how we can solve steady problems, turning to a specific time of problem known as a mean exit time problem, defined by 

```math
\begin{equation*}
\begin{array}{rcll}
D\boldsymbol{\nabla}^2 T & = & -1,
\end{array}
\end{equation*}
```

where the boundary conditions could be either absorbing, reflecting, or a combination of the two, and $T$ is the mean exit time of a particle with diffusivity $D$ out of the given domain. We cover many examples, reproducing some of the work in my papers at https://iopscience.iop.org/article/10.1088/1367-2630/abe60d and https://iopscience.iop.org/article/10.1088/1751-8121/ac4a1d.