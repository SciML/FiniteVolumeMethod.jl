"""
    get_dae_initial_condition(prob::Union{FVMProblem, FVMSystem}, num_constraints)

Given a `prob <: AbstractFVMProblem`, returns the initial condition vectors to be used 
for a differential-algebraic equation with `num_constraints` constraints. The form 
of these vectors depends on whether `prob` is a `FVMProblem` or a `FVMSystem`.

# FVMProblem 
If `prob` is a `FVMProblem`, then the returned values are:
- `u0`

These are the initial conditions from `prob`, together with `num_constraints` zeros 
padded at the end.
- `du0`

These are the initial values for the differential component, evaluated using 
[`fvm_eqs!`](@ref) with `t=prob.initial_time`. The remaining `num_constraints`
components are all zero.

# FVMSystem 
In case `prob` is a `FVMSystem`, then the returned values are similar as above 
but with some important differences. We let `n` be the number of nodes in the 
underlying mesh, and `N` the number of equations in the system.
- `u0`

These are also the initial conditions from `prob`, except the initial condition 
has been flattened into a vector of length `n * N`. The remaining `num_constraints`
components are all zero.
- `du0`

Similar to above, except the initial values for the differential component are
flattened into a vector of length `n * N`. The last `num_constraints` components
are all zero.
"""
get_dae_initial_condition

function get_dae_initial_condition(prob::FVMProblem, num_constraints)
    u0 = copy(prob.initial_condition)
    p = (prob = prob, parallel = Val(false))
    t0 = prob.initial_time
    du0 = similar(u0)
    fvm_eqs!(du0, u0, p, t0)
    append!(u0, fill(zero(eltype(u0)), num_constraints))
    append!(du0, fill(zero(eltype(du0)), num_constraints))
    return u0, du0
end
function get_dae_initial_condition(prob::FVMSystem{N}, num_constraints) where {N} 
    u0 = copy(prob.initial_condition)
    p = (prob = prob, parallel = Val(false))
    t0 = prob.initial_time
    du0 = similar(u0)
    fvm_eqs!(du0, u0, p, t0)
    flattened_u0 = vec(u0)
    flattened_du0 = vec(du0)
    append!(flattened_u0, fill(zero(eltype(u0)), num_constraints))
    append!(flattened_du0, fill(zero(eltype(du0)), num_constraints))
    return flattened_u0, flattened_du0
end 

"""
    get_differential_vars(prob::Union{FVMProblem, FVMSystem}, num_constraints)

Returns a vector whose `i`th element is `true` if the `i`th variable is a differential
variable, and `false` otherwise. The length of the vector depends on whether 
`prob` is a `FVMProblem` or a `FVMSystem`.

# FVMProblem 
If `n` is the number of points in the underlying triangulation, then the vector `v` is 
length `n + num_constraints` with `v[1:n]` all `true`, and `v[(n+1):end]` all false. 

# FVMSystem 
If `n` is the number of points in the underlying triangulation, and `N` is the number
of equations in the system, then the vector `v` is length `n * N + num_constraints` with
`v[1:(n*N)]` all `true`, and `v[(n*N+1):end]` all `false`.
"""
function get_differential_vars(prob, num_constraints)
    n = length(prob.initial_condition)
    v1 = fill(true, n)
    v2 = fill(false, num_constraints)
    append!(v1, v2)
    return v1
end

"""
    compute_flux(prob::AbstractFVMProblem, i, j, u, t)

Given an edge with indices `(i, j)`, a solution vector `u`, a current time `t`, 
and a problem `prob`, computes the flux `∇⋅q(x, y, t, α, β, γ, p) ⋅ n`, 
where `n` is the normal vector to the edge, `q` is the flux function from `prob`,
`(x, y)` is the midpoint of the edge, `(α, β, γ)` are the shape function coefficients, 
and `p` are the flux parameters from `prob`. If `prob` is an [`FVMSystem`](@ref), the returned 
value is a `Tuple` for each individual flux. The normal vector `n` is a clockwise rotation of 
the edge, meaning pointing right of the edge `(i, j)`.
"""
function compute_flux(prob::AbstractFVMProblem, i, j, u, t)
    tri = prob.mesh.triangulation
    p, q = get_point(tri, i, j)
    px, py = getxy(p)
    qx, qy = getxy(q)
    ex, ey = qx - px, qy - py
    ℓ = norm((ex, ey))
    nx, ny = ey / ℓ, -ex / ℓ
    k = get_adjacent(tri, j, i) # want the vertex in the direction of the normal
    if DelaunayTriangulation.is_boundary_index(k)
        k = get_adjacent(tri, i, j)
    else
        i, j = j, i
    end
    T, props = _safe_get_triangle_props(prob, (i, j, k))
    α, β, γ = get_shape_function_coefficients(props, T, u, prob)
    mx, my = (px + qx) / 2, (py + qy) / 2
    qv = eval_flux_function(prob, mx, my, t, α, β, γ)
    if is_system(prob)
        qn = ntuple(_neqs(prob)) do var
            local qvx, qvy
            qvx, qvy = getxy(qv[var])
            return nx * qvx + ny * qvy
        end
        return qn
    else
        qvx, qvy = getxy(qv)
        return nx * qvx + ny * qvy
    end
end
