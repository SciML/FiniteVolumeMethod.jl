# Mathematical Details

We now describe the mathematical details involved with the finite volume method as we have applied it. We assume that we have some triangulation of $\Omega$, like a (constrained) Delaunay triangulation $\mathcal D\mathcal T(\Omega)$. 

## Interior discretisation

This triangulation is used to define control volumes around each point. This is illustrated in the following figure, where (a) shows the domain $\Omega$ and its triangulation $\mathcal T(\Omega)$, together with the boundary $\partial\Omega$ shown in blue. (b) shows the mesh in (a) along with the dual mesh shown in blue, with red points showing the centroids of each triangle in $\mathcal{T}(\Omega)$. The blue polygons around each nodal point are the control volumes, and we denote the control volume around some point $\boldsymbol{x}_i$ by $\Omega_i$ and its boundary is $\partial\Omega_i$. (Note that this is the so-called ``vertex-centred approach'' to the finite volume method.)

![Dual mesh](https://github.com/DanielVandH/FiniteVolumeMethod.jl/blob/main/test/figures/triangulation_example.png?raw=true)

To be more precise, consider some interior point $\boldsymbol{x}\_{i}  = (x\_i, y\_i)^{\mathsf T} \in \Omega$ which is a point in $\mathcal T(\Omega)$, i.e. one of the black points on the figure above. We take the centroids of the neighbouring triangles of $\boldsymbol{x}\_i$ and connect these centroids to the midpoints of the associated triangle. These connections defined a closed polygon around $\boldsymbol{x}\_i$ which we denote by $\partial\Omega_i$, and its interior will be $\Omega\_i$ with some volume $V\_i$. This polygon will be comprised of a set of edges $\mathcal E_i$, and for each $\boldsymbol{x}\_{\sigma} \in\mathcal E\_i$ there will be a length $L\_\sigma$, a midpoint $\boldsymbol{x}\_{\sigma}$, and a unit normal vector $\hat{\boldsymbol{n}}\_{i, \sigma}$ which is normal to $\sigma$ and directed outwards to $\Omega\_i$ with $\|\hat{\boldsymbol{n}}\_{i,\sigma}\| = 1$. This notation is elucidated in the figure below. In this figure, the green area shows $\Omega_i$, and its boundary $\partial\Omega_i$ is in blue. The edge $\sigma$ is shown in blue. Lastly, the cyan points show an example ordering $(v_{k1}, v_{k2}, v_{k3})$ of a triangle $T_k \in \mathcal T(\Omega)$. It is with these control volumes that we can now discretise our PDE $\partial_t u + \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{q} = R$.

![Control volume](https://github.com/DanielVandH/FiniteVolumeMethod.jl/blob/main/test/figures/control_volume_example.png?raw=true)

Let us start by integrating our equations around $\Omega\_i$ and moving the time-derivative outside the integral:

```math
\begin{equation}
\dfrac{\mathrm d}{\mathrm dt}\iint_{\Omega_i} u~ \mathrm{d}V + \iint_{\Omega_i} \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{q}~ \mathrm{d}V = \iint_{\Omega_i} R~ \mathrm{d}V. 
\end{equation} 
```

Using the divergence theorem, the second integral becomes

```math
\begin{equation}
\iint_{\Omega_i}  \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{q}~\mathrm{d}V = \oint_{\partial\Omega_i} \boldsymbol{q} \boldsymbol{\cdot} \hat{\boldsymbol{n}}_{i, \sigma}~\mathrm{d}s = \sum_{\sigma\in \mathcal E_i} \int_\sigma \boldsymbol{q} \boldsymbol{\cdot} \hat{\boldsymbol{n}}_{i, \sigma}~\mathrm{d}s,
\end{equation}
```

where the last equality follows from integrating over each individual line segment that defines $\partial\Omega_i$. We now define the control volume averages,

```math
\begin{equation}
\bar u_i = \frac{1}{V_i}\iint_{\Omega_i} u~\mathrm dV, \qquad \bar R_i = \frac{1}{V_i}\iint_{\Omega_i} R~\mathrm dV,
\end{equation}
```

so that our integral formulation becomes

```math
\begin{equation}
\dfrac{\mathrm d\bar u_i}{\mathrm dt} + \frac{1}{V_i}\sum_{\sigma\in\mathcal E_i}\int_{\sigma} \boldsymbol{q} \boldsymbol{\cdot} \hat{\boldsymbol{n}}_{i, \sigma}~\mathrm ds = \bar R_i. 
\end{equation}
```

The line integrals can be approximated using a midpoint rule, 

```math
\begin{equation}
\int_\sigma \boldsymbol{q} \boldsymbol{\cdot} \hat{\boldsymbol{n}}\_{i, \sigma}~\mathrm ds \approx \left[\boldsymbol{q}(x\_\sigma, y\_\sigma, t, u) \boldsymbol{\cdot} \hat{\boldsymbol{n}}\_{i, \sigma}\right] L_\sigma. 
\end{equation} 
```

Lastly, the control volume averages can be approximated by simply replacing them by their value at $(x\_i, y\_i)$. We thus obtain the following approximation, where $\tilde u\_i$ denotes an approximation to the exact solution $u$ at $(x_i, y_i)$:

```math 
\begin{equation} 
\frac{\mathrm d\tilde u_i}{\mathrm dt} + \frac{1}{V_i}\sum_{\sigma \in \mathcal E_i} \left[\boldsymbol{q}(x_\sigma, y_\sigma, t, u) \boldsymbol{\cdot} \hat{\boldsymbol{n}}\_{i, \sigma}\right] L\_\sigma = \tilde R_i,
\end{equation} 
```

where $\tilde R\_i$ is the approximation to $\bar R\_i$ at $(x\_i, y\_i)$. This approximation is what we use in the interior of $\Omega$ for approximating the value of $u$ at each node. 

We still need to discuss how we compute $\boldsymbol{q}(x\_{\sigma}, y\_{\sigma}, t, u)$. To deal with this function, let $\mathcal T\_i$ be the set of triangles in $\mathcal T(\Omega)$ that have $\boldsymbol{x}\_i$ as a node, and consider a triangle $T\_k \in \mathcal T\_i$. We will inteprolate $\tilde u$ with a linear shape function in $T_k$, so that 

```math 
\begin{equation}
\tilde u(x, y, t) = \alpha_k x + \beta_ky + \gamma_k, \quad (x, y) \in T_k.
\end{equation} 
```

(The dependence of the coefficients $\alpha\_k$, $\beta\_k$, and $\gamma\_k$ on $t$ is suppressed.) We suppose that the three nodes defining $T_k$ are $(v\_{k1}, v\_{k2}, v\_{k3})$, and these points are given in a counter-clockwise order. An example of how these points are defined was shown in the control volume schematic figure. We can find $\alpha\_k$, $\beta\_k$, and $\gamma\_k$ by noting that we know the values of $\tilde u$ at the nodes $v\_{k1}$, $v\_{k2}$, and $v\_{k3}$, either from the initial condition or from the previous time-step of the integration. We will denote the value of $\tilde u$ at $v\_{ki}$ by $\tilde u\_{v_{ki}}$, $i=1,2,3$. We then have the system of equations, 

```math 
\begin{equation} 
\tilde u_{v_{ki}} = \alpha_k x_{v_{ki}} + \beta_k y_{v_{ki}} + \gamma_k, \quad i=1,2,3, 
\end{equation} 
```

where $x_{v_{ki}}$ and $y_{v_{ki}}$ denote the $x$- and $y$-coordinates of the point $v\_{ki}$, respectively, for $i=1,2,3$. The system can be written in matrix form as 

```math
\begin{bmatrix} 
x_{v_{k1}} & y_{v_{k1}} & 1 \\ x_{v_{k2}} & y_{v_{k2}} & 1 \\ x_{v_{k3}} & y_{v_{k3}} & 1 \end{bmatrix}\begin{bmatrix} \alpha_k \\ \beta_k \\ \gamma_k \end{bmatrix} = \begin{bmatrix} \tilde u_{v_{k1}} \\ \tilde u_{v_{k2}} \\ \tilde u_{v_{k3}}
\end{bmatrix} 
```

Using Cramer's rule, we can define 

```math
\begin{equation} 
\begin{array}{lcllcllcl}
s_{k1} &=& \dfrac{y_{v_{k2}} - y_{v_{k3}}}{\Delta_k}, & s_{k2} &=& \dfrac{y_{v_{k3}}-y_{v_{k1}}}{\Delta_k}, & s_{k3} &=& \dfrac{y_{v_{k1}}-y_{v_{k2}}}{\Delta_k}, \\
s_{k4} &=& \dfrac{x_{v_{k3}}-x_{v_{k2}}}{\Delta_k},& s_{k5} &=& \dfrac{x_{v_{k1}}-x_{v_{k3}}}{\Delta_k}, & s_{k6} &=& \dfrac{x_{v_{k2}}-x_{v_{k1}}}{\Delta_k}, \\
s_{k7} &=& \dfrac{x_{v_{k2}}y_{v_{k3}}-x_{v_{k3}}y_{v_{k2}}}{\Delta_k}, & s_{k8} &=& \dfrac{x_{v_{k3}}y_{v_{k1}}-x_{v_{k1}}y_{v_{k3}}}{\Delta_k},& s_{k9} &= &\dfrac{x_{v_{k1}}y_{v_{k2}}-x_{v_{k2}}y_{v_{k1}}}{\Delta_k}, 
\end{array} 
\end{equation} 
```

and 

```math 
\begin{equation}
\Delta_k = x_{v_{k1}}y_{v_{k2}}-x_{v_{k2}}y_{v_{k1}}-x_{v_{k1}}y_{v_{k3}}+x_{v_{k3}}y_{v_{k1}}+x_{v_{k2}}y_{v_{k3}}-x_{v_{k3}}y_{v_{k2}}.
\end{equation}
```

With this notation, 

```math
\begin{equation} 
\begin{array}{rcl}
\alpha_k &=& s_{k1}\tilde u_{v_{k1}} + s_{k2}\tilde u_{v_{k2}} + s_{k3}\tilde u_{v_{k3}}, \\
\beta_k &= & s_{k4}\tilde u_{v_{k1}} + s_{k5}\tilde u_{v_{k2}} + s_{k6}\tilde u_{v_{k3}}, \\
\gamma_k &=& s_{k7}\tilde u_{v_{k1}} + s_{k8}\tilde u_{v_{k2}} + s_{k9}\tilde u_{v_{k3}}.
\end{array}
\end{equation} 
```

With these coefficients, our approximation becomes 

```math 
\begin{equation} 
\frac{\mathrm du_i}{\mathrm dt} + \frac{1}{V_i}\sum_{\sigma\in\mathcal E_i} \left[\boldsymbol{q}\left(x_\sigma, y_\sigma, t, \alpha_{k(\sigma)}x_\sigma + \beta_{k(\sigma)}y_\sigma + \gamma_{k(\sigma)}\right) \boldsymbol{\cdot} \hat{\boldsymbol{n}}\_{i, \sigma}\right] L\_\sigma = R_i,
\end{equation} 
```

where we now drop the tilde notation and make the approximations implicit, and now the $k(\sigma)$ notation is used to refer to the edge $\sigma$ inside triangle $T_{k(\sigma)}$. This linear shape function also allows to compute gradients like $\boldsymbol{\nabla} u(x_\sigma, y_\sigma)$, since $\boldsymbol{\nabla} u(x_\sigma, y_\sigma) = (\alpha_{k(\sigma)}, \beta_{k(\sigma)})^{\mathsf T}$.

## Boundary conditions 

As discussed at the start, we only support boundary conditions of the form 

```math
\begin{array}{rcl}
\boldsymbol{q}(x, y, t, u) \boldsymbol{\cdot} \hat{\boldsymbol{n}}(x, y) = 0, \\
\mathrm du(x, y, t)/\mathrm dt = a(x, y, t, u), \\
u(x, y, t) = a(x, y, t, u),
\end{array} \quad (x, y)^{\mathsf T} \in \partial\Omega.
```

For the Neumann boundary condition, recall that the integral form of our PDE was 

```math
\begin{equation}
\dfrac{\mathrm d\bar u_i}{\mathrm dt} + \frac{1}{V_i}\sum_{\sigma\in\mathcal E_i}\int_{\sigma} \boldsymbol{q} \boldsymbol{\cdot} \hat{\boldsymbol{n}}_{i, \sigma}~\mathrm ds = \bar R_i. 
\end{equation}
```

Thus, if $\sigma$ is an edge such that $\boldsymbol{q} \boldsymbol{\cdot} \hat{\boldsymbol{n}} = 0$, then the contribution from $\sigma$ to the above sum is zero. Thus, in our code, we simply skip over such a $\sigma$ when computing the sum. 

For the time-dependent Dirichlet boundary condition, we can skip over nodes with this condition and simply set $\mathrm du\_i/\mathrm dt = a(x, y, t, u)$. 

Lastly, for the Dirichlet boundary conditions, we leave $\mathrm du\_i/\mathrm dt = 0$ and simply update the value of $u\_i$ with $a(x, y, t, u\_i)$ at the end of each iteration. This is done using callbacks.
