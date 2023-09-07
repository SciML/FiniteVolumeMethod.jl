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

# Heat Convection on a Square Plate with Inhomogeneous Neumann Boundary Conditions and a Robin Boundary Condition

This [tutorial](heat_convection_on_a_square_plate_with_inhomogeneous_neumann_boundary_conditions_and_a_robin_boundary_condition.md) considers the following heat condition problem:
```math
\begin{equation*}
\begin{aligned}
\frac{1}{\alpha}\pdv{T}{t} &= \pdv[2]{T}{x} + \pdv[2]{T}{y} & 0 < x < L,\, 0 < y < L, \\[6pt]
\pdv{T}{x} &= 0 & x=0,\,x=L, \\[6pt]
\pdv{T}{y} &= -\frac{q}{k} & y=0, \\[6pt]
k\pdv{T}{y} + hT &= hT_{\infty} & y=L,\\[6pt] 
T(x, y, 0) &= T_0.
\end{aligned}
\end{equation*}
```
This problem has an exact solution given by:[^6]
```math
T(x, y, t) = -\frac{q}{k}y + T_{\infty} + q\left(\frac{1}{h}+\frac{L}{k}\right) + \sum_{n=1}^\infty A_n\cos\left(\beta_ny\right)\mathrm{e}^{-\alpha\beta_n^2t},
```
where $\tan(\beta_nL) = h\beta_n/k$ and
```math
A_n = \frac{2}{L}\int_0^L \left[T_0 + \frac{q}{k}y - T_{\infty} - q\left(\frac{1}{h}+\frac{L}{k}\right)\right]\cos(\beta_ny) \dd{y},
```
though as in the previous tutorials we do not reference this exact solution in the tutorial.

[^6]: This exact solution can be derived using separation of variables as follows.

    First, note that the solution is constant in $x$ along the left and right boundaries, and the boundary conditions and initial conditions both do not involve $x$ directly. We thus seek a solution that is independent in $x$ when we consider applying separation of variables.

    Before we can apply separation of variables, let us first make the problem homogeneous, which requires that we first find the steady state solution $T_e$. Since the problem is independent in $x$, the steady state simply requires $T_e''(y) = 0$, or $T_e(y) = Ay + B$. The boundary conditions $T_e'(0) = -q/k$ and $kT_e'(L) + hT_e(L) = hT_{\infty}$ then give 
    ```math
    A = -\frac{q}{k}, \quad B = T_{\infty} + q\left(\frac{1}{h} + \frac{L}{k}\right),
    ```
    so that 
    ```math
    \begin{equation}\label{eq:sepvareqlb}
    T_e(y) = -\frac{q}{k}y +T_{\infty} + q\left(\frac{1}{h} + \frac{L}{k}\right).
    \end{equation}
    ```

    With this equilibrium solution \eqref{eq:sepvareqlb} we define $U(y, t) = T(x, y, t) - T_e(y)$ which will also be independent of $x$, or $T(x, y, t) = U(y, t) + T_e(y)$. With this definition, our PDE becomes, where we also use the $x$-invariance:
    ```math
    \begin{equation}\label{eq:newsepvarpdeh}
    \begin{aligned}
    \frac{1}{\alpha}\pdv{U}{t} &= \pdv[2]{U}{y} & 0<y<L,\\[6pt] 
    \pdv{U}{y} &= 0 & y=0,\\[6pt] 
    k\pdv{U}{y} + hU &= 0 & y=L,\\[6pt] 
    U(y, 0) &= T_0 - T_e(y).
    \end{aligned}
    \end{equation}
    ```

    We now solve \eqref{eq:newsepvarpdeh} using separation of variables. Assume that $U(y, t) = Y(t)\mathcal T(t)$. Then 
    ```math
    \frac{1}{\alpha}\frac{\mathcal T'}{\mathcal T} = \frac{Y''}{Y} = -\lambda,
    ```
    for some constant $\lambda \geq 0$. 

    For the case $\lambda = 0$: Here, $Y'' = 0$ so $Y = A + By$. Since $Y'(0) = 0$, this gives $B=0$, and thus $Y = A$. Then, since $kY'(L) + hY(L) = 0$, we get $hY(L) = 0$ or $A = 0$ (since $h \neq 0$). So, $\lambda=0$ is not an eigenvalue as $Y=0$ is a zero function.

    For the case $\lambda > 0$: Here, write $\lambda=\beta^2$ so that $Y''/Y = -\beta^2$ and thus $Y = A\cos(\beta y) + B\sin(\beta y)$, giving $Y' = -A\beta\sin(\beta y) + B\beta\cos(\beta y)$. The bonudary condition $Y'(0) = 0$ gives $B\beta=0$ and thus $B=0$, so $Y = A\cos(\beta y)$. Then, $kY'(L) + hY(L) = 0$ gives
    ```math
    \begin{align*}
    h\cos(L\beta) &= \beta k \sin(L\beta) \\
    \tan(L\beta) &= \frac{h}{k}\beta.
    \end{align*}
    ```
    Thus, the $\beta$s are the solutions to the transcendental equation $\tan(L\beta) = h\beta/k$. So, $\lambda_n = \beta_n^2$ for $n=1,2,3,\ldots$. Returning to $\mathcal T'/(\alpha \mathcal T) = -\lambda$, we see that 
    ```math
    \mathcal T_n(t) = \mathrm{e}^{-\alpha\beta_n^2t}.
    ```
    So, putting everything together, the solution takes the form
    ```math
    U(y, t) = \sum_{n=1}^\infty A_n\left(\beta_ny\right)\mathrm{e}^{-\alpha\beta_n^2t},
    ```
    where $\tan(\beta_nL) = h\beta_n/k$ and $A_n$ is determined by the initial condition via
    ```math
    A_n = \frac{2}{L}\int_0^L \left[T_0 - T_e(y)\right]\cos(\beta_ny) \dd{y}.
    ```
    This gives the exact solution $T(x, y, t) = U(y, t) + T_e(y)$.