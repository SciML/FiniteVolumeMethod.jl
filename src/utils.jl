function _safe_get_triangle_props(prob::AbstractFVMProblem, T)
    i, j, k = indices(T)
    props = prob.mesh.triangle_props
    if haskey(props, (i, j, k))
        return (i, j, k), get_triangle_props(prob, i, j, k)
    elseif haskey(props, (j, k, i))
        return (j, k, i), get_triangle_props(prob, j, k, i)
    else
        return (k, i, j), get_triangle_props(prob, k, i, j)
    end
end

"""
    pl_interpolate(prob, T, u, x, y)

Given a `prob <: AbstractFVMProblem`, a triangle `T` containing a point `(x, y)`, 
and a set of function values `u` at the corresponding nodes of `prob`, interpolates 
the solution at the point `(x, y)` using piecewise linear interpolation.
"""
function pl_interpolate(prob, T, u, x, y)
    T, props = _safe_get_triangle_props(prob, T)
    α, β, γ = get_shape_function_coefficients(props, T, u, prob)
    return α .* x .+ β .* y .+ γ
end