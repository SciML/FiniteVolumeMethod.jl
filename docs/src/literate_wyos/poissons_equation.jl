# # Poisson's Equation 
# ```@contents 
# Pages = ["poissons_equation.md"]
# ``` 
# We now write a solver for Poisson's equation. What we produce 
# in this section can also be accessed in `FiniteVolumeMethod.PoissonsEquation`.

# ## Mathematical Details
# We start by describing the mathematical details. The problems we will be solving 
# take the form 
# ```math 
# \grad^2 u = f(\vb x).
# ``` 
# Note that this is very similar to a mean exit time problem with constant $D$, 
# where $f(\vb x) = -1/D$. The mathematical details are thus also very similar. 
# In particular, we already know that 
# ```math 
# \frac{1}{V_i}\sum_{\sigma\in\mathcal E_i} D(\vb x_\sigma)\left[\left(s_{k, 11}n_\sigma^x+s_{k,21}n_\sigma^y\right)T_{k1} + \left(s_{k,12}n_\sigma^x+s_{k,22}n_\sigma^y\right)T_{k2}+\left(s_{k,13}n_\sigma^x+s_{k,23}n_\sigma^y\right)T_{k3}\right]L_\sigma = f(\vb x_i),
# ```
# where 