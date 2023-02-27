# FiniteVolumeMethod

[![DOI](https://zenodo.org/badge/561533716.svg)](https://zenodo.org/badge/latestdoi/561533716)

This is a package for solving partial differential equations (PDEs) of the form 

```math
\dfrac{\partial u(x, y, t)}{\partial t} + \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{q}(x, y, t, u) = R(x, y, t, u), \quad (x, y)^{\mathsf T} \in \Omega \subset \mathbb R^2,t>0,
```

or

```math
\boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{q}(x, y, t, u) = R(x, y, t, u), \quad (x, y)^{\mathsf T} \in \Omega \subset \mathbb R^2,t>0,
```


with flux and reaction functions $\boldsymbol{q}$ and $R$, respectively, using the finite volume method. The boundary conditions are assumed to take on any of the three forms:

```math
\begin{array}{rcl}
\boldsymbol{q}(x, y, t, u) \boldsymbol{\cdot} \hat{\boldsymbol{n}}(x, y) = 0, \\
\mathrm du(x, y, t)/\mathrm dt = a(x, y, t, u), \\
u(x, y, t) = a(x, y, t, u),
\end{array} \quad (x, y)^{\mathsf T} \in \partial\Omega.
```

This first condition is a *homogeneous Neumann* boundary condition, letting $\hat{\boldsymbol{n}}(x, y)$ be the unit outward normal vector field on $\partial\Omega$ (the boundary of $\Omega$); it is possible to extend this to the inhomogeneous case, it just has not been done yet. The second condition is a *time-dependent Dirichlet* condition, and the last condition is a *Dirichlet* condition. 

An interface is also provided for solving equations of the form

```math
\frac{\partial u(x, y, t)}{\partial t} = \boldsymbol{\nabla} \boldsymbol{\cdot} \left[T(x, y, t, u)D(x, y, t, u)\boldsymbol{\nabla} u(x, y, t)\right] + T(x, y, t, u)R(x, y, t, u),
```

and similarly if $\partial u/\partial t = 0$, where $T$ is called the *delay function*, $D$ the *diffusion function*, and $R$ the *reaction function*; the same delay is assumed to scale both diffusion and reaction. The conversion is done by noting that the corresponding flux function $\boldsymbol{q} = (q_1, q_2)^{\mathsf T}$ is simply $q_i(x, y, t, u) = -T(x, y, t, u)D(x, y, t, u)g_i$, $i=1,2$, where $(g_1, g_2)^{\mathsf T} \equiv \boldsymbol{\nabla}u(x, y, t)$ (gradients are approximated using linear interpolants; more on this in the Mathematical Details section). Similarly, the reaction function is modified so that $\tilde{R}(x, y, t, u) = T(x, y, t, u)R(x, y, t, u)$.