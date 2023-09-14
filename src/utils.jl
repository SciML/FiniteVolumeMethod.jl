function _safe_get_triangle_props(mesh::FVMGeometry, T)
    i, j, k = indices(T)
    props = mesh.triangle_props
    if haskey(props, (i, j, k))
        return (i, j, k), get_triangle_props(mesh, i, j, k)
    elseif haskey(props, (j, k, i))
        return (j, k, i), get_triangle_props(mesh, j, k, i)
    else
        return (k, i, j), get_triangle_props(mesh, k, i, j)
    end
end
_safe_get_triangle_props(prob::AbstractFVMProblem, T) = _safe_get_triangle_props(prob.mesh, T)

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

"""
    two_point_interpolant(mesh, u, i, j, mx, my)

Given a `mesh <: FVMGeometry`, a set of function values `u` at the nodes of `mesh`, 
and a point `(mx, my)` on the line segment between the nodes `i` and `j`,
interpolates the solution at the point `(mx, my)` using two-point interpolation.
"""
function two_point_interpolant(mesh, u::AbstractVector, i, j, mx, my)
    xᵢ, yᵢ = get_point(mesh, i)
    xⱼ, yⱼ = get_point(mesh, j)
    ℓ = sqrt((xⱼ - xᵢ)^2 + (yⱼ - yᵢ)^2)
    ℓ′ = sqrt((mx - xᵢ)^2 + (my - yᵢ)^2)
    return u[i] + (u[j] - u[i]) * ℓ′ / ℓ
end
