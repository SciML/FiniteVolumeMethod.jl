# Section Overview 

We provide many tutorials in this section. Each tutorial is self-contained, although 
more detail will be offered in the earlier examples. At the end of each tutorial we show the uncommented code, but if you want to see the actual script itself that generates these tutorials, then click on the `Edit on GitHub` section on the top right of the respective tutorial.

We list all the examples below, but the tutorials can be accessed in their respective sections of the sidebar.

# Diffusion Equation on a Square Plate
[This tutorial](diffusion_equation_on_a_square_plate.md) considers a diffusion equation problem on a square plate:

```math
\begin{equation*}
\begin{aligned}
\pdv{u(\vb x, t)}{t} &= \frac{1}{9}\grad^2 u(\vb x, t)  & \vb x \in \Omega,\,t>0, \\[6pt]
u(\vb x, t) & =  0  &\vb x \in \partial\Omega,\,t>0,\\[6pt]
u(\vb x, 0) &= f(\vb x) & \vb x \in \Omega,
\end{aligned}
\end{equation*}
```
where $\Omega = [0, 2]^2$ and
```math
f(x, y) = \begin{cases} 50 & y \leq 1, \\ 0 & y > 1. \end{cases}
```

This problem actually has an exact solution, given by[^1]

```math
u(\vb x, t) = \frac{200}{\pi^2}\sum_{m=1}^\infty\sum_{n=1}^\infty \frac{\left[1 + (-1)^{m+1}\right]\left[1-\cos\left(\frac{n\pi}{2}\right)\right]}{mn}\sin\left(\frac{m\pi x}{2}\right)\sin\left(\frac{n\pi y}{2}\right)\mathrm{e}^{-\frac{1}{36}\pi^2(m^2+n^2)t},
```

although we do not refer to this in the tutorial (only in the tests).

[^1]: See [here](http://ramanujan.math.trinity.edu/rdaileda/teach/s12/m3357/lectures/lecture_3_6_short.pdf) for a derivation.

# Diffusion Equation in a Wedge with Mixed Boundary Conditions 
This [tutorial](diffusion_equation_in_a_wedge_with_mixed_boundary_conditions.md) considers a similar example as in the first example, except now the diffusion is in a wedge, has mixed boundary conditions, and is also now in polar coordinates:

```math
\begin{equation*}
\begin{aligned}
\pdv{u(r, \theta, t)}{t} &= \grad^2u(r,\theta,t), & 0<r<1,\,0<\theta<\alpha,\,t>0,\\[6pt]
\pdv{u(r, 0, t)}{\theta} & = 0 & 0<r<1,\,t>0,\\[6pt]
u(1, \theta, t) &= 0 & 0<\theta<\alpha,\,t>0,\\[6pt]
\pdv{u(r,\alpha,t)}{\theta} & = 0 & 0<\theta<\alpha,\,t>0,\\[6pt]
u(r, \theta, 0) &= f(r,\theta) & 0<r<1,\,0<\theta<\alpha,
\end{aligned}
\end{equation*}
```
where we take $f(r,\theta) = 1-r$ and $\alpha=\pi/4$. This problem also has an exact solution:[^2]

```math
u(r, \theta, t) = \frac{1}{2}\sum_{m=1}^\infty A_{0, m}\mathrm{e}^{-\zeta_{0, m}^2t}J_0(\zeta_{0, m}r) + \sum_{n=1}^\infty\sum_{m=1}^\infty A_{n, m}\mathrm{e}^{-\zeta_{n,m}^2t}J_{n\pi/\alpha}\left(\zeta_{n\pi/\alpha,m}r\right)\cos\left(\frac{n\pi\theta}{\alpha}\right),
```
where
```math
A_{n, m} = \frac{4}{\alpha J_{n\pi/\alpha+1}^2\left(\zeta_{n\pi/\alpha,m}\right)}\int_0^1\int_0^\alpha f(r, \theta)J_{n\pi/\alpha}\left(\zeta_{n\pi/\alpha,m}r\right)\cos\left(\frac{n\pi\theta}{\alpha}\right) r\dd{r}\dd{\theta}
```
for $n=0,1,2,\ldots$ and $m=1,2,3,\ldots$, and where we write the roots of $J_\mu$, the $\mu$th order Bessel function of the first kind, as $0 < \zeta_{\mu, 1} < \zeta_{\mu, 2} < \cdots$ with $\zeta_{\mu, m} \to \infty$ as $m \to \infty$. We don't discuss this in the tutorial, but it is used in the tests.

[^2]: To derive this, use $u(r, \theta, t) = \mathrm{e}^{-\lambda t}v(r, \theta)$ and use separation of variables.

# Reaction Diffusion Equation with a Time-dependent Dirichlet Boundary Condition on a Disk 
This [tutorial](reaction_diffusion_equation_with_a_time_dependent_dirichlet_boundary_condition_on_a_disk.md) considers a reaction-diffusion problem with a $\mathrm du/\mathrm dt$ condition on a disk in polar coordinates:
```math
\begin{equation*}
\begin{aligned}
\pdv{u(r, \theta, t)}{t} &= \div[u\grad u] + u(1-u) & 0<r<1,\,\theta<2\pi,\\[6pt]
\dv{u(1, \theta, t)}{t} &= u(1,\theta,t) & 0<\theta<2\pi,\,t>0,\\[6pt]
u(r,\theta,0) &= \sqrt{I_0(\sqrt{2}r)} & 0<r<1,\,0<\theta<2\pi,
\end{aligned}
\end{equation*}
```
where $I_0$ is the modified Bessel function of the first kind of order zero. This problem also has an exact solution,[^3]
```math
u(r, \theta, t) = \mathrm{e}^t\sqrt{I_0(\sqrt{2}r)}
```
that we use in the tests.

[^3]: See [Bokhari et al. (2008)](https://doi.org/10.1016/j.na.2007.11.012) for a derivation.

# Porous-Medium Equation 
This [tutorial](porous_medium_equation.md) considers the porous-medium equation, given by 

```math
\begin{equation}\label{eq:porousmedium}
\pdv{u}{t} = D\div [u^{m-1}\grad u],
\end{equation}
```

with initial condition $u(\vb x, 0) = M\delta(\vb x)$ where $\delta(\vb x)$ is the Dirac delta function and $M = \iint_{\mathbb R^2} u(\vb x, t)\dd{A}$. This problem also has an exact solution that we use for defining an approximation to $\mathbb R^2$ when solving this numerically:[^4]

```math
\begin{equation}\label{eq:porousmediumexact}
u(\vb x, t) = \begin{cases} (Dt)^{-1/m}\left[\left(\frac{M}{4\pi}\right)^{(m-1)/m}-\frac{m-1}{4m}\left(x^2+y^2\right)(Dt)^{-1/m}\right]^{1/(m-1)} & x^2+y^2 < R_{m, M}(Dt), \\
0 & x^2+y^2 \geq R_{m, M}(Dt), \end{cases}
\end{equation}
```

where $R_{m, M} = [4m/(m-1)][M/(4\pi)]^{(m-1)/m}$.

[^4]: This exact solution is derived in Section 17.5 of the book _The porous medium equation: Mathematical theory_ by J. L. Vázquez (2007).

We also consider a similar problem to \eqref{eq:porousmedium}, where now the problem has a linear source:

```math
\pdv{u}{t} = D\div [u^{m-1}\grad u] + \lambda u, \quad \lambda > 0,
```

which has an exact solution given by

```math
u(\vb x, t) = \mathrm{e}^{\lambda t}v\left(\vb x, \frac{D}{\lambda(m-1)}\left[\mathrm{e}^{\lambda(m-1)t}-1\right]\right),
```

where $v$ is the exact solution from \eqref{eq:porousmediumexact} except with $D=1$.

# Porous-Fisher Equation and Travelling Waves 

This [tutorial](porous_fisher_equation_and_travelling_waves.md) looks at the Porous-Fisher equation,
```math
\begin{equation*}
\begin{aligned}
\pdv{u(\vb x, t)}{t} &= D \div[u\grad u] + \lambda u(1-u) & 0<x<a,\,0<y<b,\,t>0,\\[6pt]
u(x, 0, t) & =  1 & 0<x<a,\,t>0,\\[6pt]
u(x, b, t) & =  0 & 0<x<a,\,t>0,\\[6pt]
\pdv{u(0, y, t)}{x} &= 0 & 0<y<b,\,t>0,\\[6pt]
\pdv{u(a, y, t)}{x} &= 0 & 0 < y < b,\,t>0,\\[6pt]
u(\vb x, 0) & = f(y) & 0 \leq x \leq a,\, 0 \leq  y\leq b.
\end{aligned}
\end{equation*}
```
We solve this problem and also compare it to known travelling wave solutions.

# Piecewise Linear and Natural Neighbour Interpolation for an Advection-Diffusion Equation

This [tutorial](piecewise_linear_and_natural_neighbour_interpolation_for_an_advection_diffusion_equation.md) looks at how we can apply interpolation to solutions from the PDEs discussed, making use of [NaturalNeighbours.jl](https://github.com/DanielVandH/NaturalNeighbours.jl) for this purpose. To demonstrate this, we use a two-dimensional advection equation of the form
```math
\pdv{u}{t} = D\pdv[2]{u}{x} + D\pdv[2]{u}{y} - \nu\pdv{u}{x},
```
defined for $\vb x \in \mathbb R^2$ and $u(\vb x, 0) = \delta(\vb x)$, where $\delta$ is the Dirac delta function, and with homogeneous 
Dirichlet conditions. This problem has an exact solution, given by[^5]
```math
u(\vb x, t) = \frac{1}{4D\pi t}\exp\left(\frac{-(x-\nu t)^2-y^2}{4Dt}\right).
```

[^5]: A derivation is given [here](https://math.stackexchange.com/a/3925070/861404).

# Helmholtz Equation with Inhomogeneous Boundary Conditions 
This [tutorial](helmholtz_equation_with_inhomogeneous_boundary_conditions.md) considers the Helmholtz equation with inhomogeneous boundary conditions:
```math
\begin{equation}
\begin{aligned}
\grad^2 u(\vb x) + u(\vb x) &= 0 & \vb x \in [-1, 1]^2 \\
\pdv{u}{\vb n} &= 1 & \vb x \in\partial[-1,1]^2.
\end{aligned}
\end{equation}
```
This is different to the other problems considered thus far as the problem is now a _steady state problem_. The exact solution is
```math
u(x, y) = -\frac{\cos(x+1)+\cos(1-x)+\cos(y+1)+\cos(1-y)}{\sin(2)}.
```

# Laplace's Equation with Internal Dirichlet Conditions 
This [tutorial](laplaces_equation_with_internal_dirichlet_conditions.md) considers Laplace's equation with internal Dirichlet conditions:
```math
\begin{equation}
\begin{aligned}
\grad^2 u &= 0 & \vb x \in [0, 1]^2, \\ 
u(0, y) &= 100y & 0 \leq y \leq 1, \\
u(1, y) &= 100y & 0 \leq y \leq 1, \\
u(x, 0) &= 0 & 0 \leq x \leq 1, \\
u(x, 1) &= 100 & 0 \leq x \leq 1, \\
u(1/2, y) &= 0 & 0 \leq y \leq 2/5.
\end{aligned}
\end{equation}
```
This last condition $u(1/2, y) = 0$ is the internal condition that needs to be dealt with.

# Equilibrium Temperature Distribution with Mixed Boundary Conditions and using EnsembleProblems

This [tutorial](equilibrium_temperature_distribution_with_mixed_boundary_conditions.md) considers the equilibrium temperature distribution in a square plate with mixed boundary conditions:
```math
\begin{equation}
\begin{aligned}
\grad^2 T &= 0 & \vb x \in \Omega, \\
\grad T \vdot \vu n &= 0 & \vb x \in \Gamma_1, \\
T &= 40 & \vb x \in \Gamma_2, \\
k\grad T \vdot \vu n &= h(T_{\infty} - T) & \vb x \in \Gamma_3, \\
T &= 70 & \vb x \in \Gamma_4. \\
\end{aligned}
\end{equation}
```
This domain $\Omega$ with boundary $\partial\Omega=\Gamma_1\cup\Gamma_2\cup\Gamma_3\cup\Gamma_4$ is shown below. For this tutorial, we also consider how varying $T_{\infty}$ affects the results, using interpolation and `EnsembleProblem`s to study this efficiently.

```@setup equilex
using CairoMakie
A = (0.0, 0.06)
B = (0.03, 0.06)
F = (0.03, 0.05)
G = (0.05, 0.03)
C = (0.06, 0.03)
D = (0.06, 0.0)
E = (0.0, 0.0)
fig = Figure(fontsize=33)
ax = Axis(fig[1, 1], xlabel="x", ylabel="y")
lines!(ax, [A, E, D], color=:red, linewidth=5)
lines!(ax, [B, F, G, C], color=:blue, linewidth=5)
lines!(ax, [C, D], color=:black, linewidth=5)
lines!(ax, [A, B], color=:magenta, linewidth=5)
text!(ax, [(0.03, 0.001)], text=L"\Gamma_1", fontsize=44)
text!(ax, [(0.055, 0.01)], text=L"\Gamma_2", fontsize=44)
text!(ax, [(0.04, 0.04)], text=L"\Gamma_3", fontsize=44)
text!(ax, [(0.015, 0.053)], text=L"\Gamma_4", fontsize=44)
text!(ax, [(0.001, 0.03)], text=L"\Gamma_1", fontsize=44)
```

```@example equilex 
fig #hide
```