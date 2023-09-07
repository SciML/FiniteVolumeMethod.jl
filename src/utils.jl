function _safe_get_triangle_props(prob::AbstractFVMProblem, T)
    i, j, k = indices(T)
    props = prob.mesh.triangle_props
    if haskey(props, (i, j, k))
        return get_triangle_props(prob, i, j, k)
    elseif haskey(props, (j, k, i))
        return get_triangle_props(prob, j, k, i)
    else
        return get_triangle_props(prob, k, i, j)
    end
end

"""
    pl_interpolate(sol, T, i, x, y)

Given a solution `sol` from a `AbstractFVMProblem`, a triangle `T` containing a point `(x, y)`,
and a time index `i`, interpolates the solution at the point `(x, y)` at time `sol.t[i]` using 
piecewise linear interpolation.
"""     
function pl_interpolate(sol, T, i, x, y)
    prob = sol.prob.p.prob
    props = _safe_get_triangle_props(prob, T)
    α, β, γ = get_shape_function_coefficients(props, T, sol.u[i], prob)
    return α .* x .+ β .* y .+ γ
end