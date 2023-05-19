"""
    eval_interpolant!(αβγ, prob::FVMProblem, x, y, T, u)
    eval_interpolant(sol, x, y, t_idx::Integer, T)
    eval_interpolant(sol, x, y, t::Number, T)

Evaluates the interpolant corresponding to the `FVMProblem` at the point `(x, y)` and time `t`
(or `sol.t[t_idx]`). 
"""
function eval_interpolant end
@inline function eval_interpolant!(αβγ, prob::FVMProblem, x, y, T, u)
    T, _ = DelaunayTriangulation.contains_triangle(get_triangulation(prob), T)
    linear_shape_function_coefficients!(αβγ, u, prob, T)
    α = getα(αβγ)
    β = getβ(αβγ)
    γ = getγ(αβγ)
    return α * x + β * y + γ
end
@inline function eval_interpolant(sol, x, y, t_idx::Integer, T)
    prob = sol.prob.p[1]
    if length(sol.prob.p) == 3 # need to handle the parallel case
        shape_coeffs = sol.prob.p[3]
        new_shape_coeffs = get_tmp(shape_coeffs, x)
    else
        shape_coeffs = sol.prob.p[4]
        new_shape_coeffs = get_tmp(first(shape_coeffs), x)
    end
    return eval_interpolant!(new_shape_coeffs, prob, x, y, T, sol.u[t_idx])
end
@inline function eval_interpolant(sol, x, y, t::Number, T)
    prob = sol.prob.p[1]
    if length(sol.prob.p) == 3 # need to handle the parallel case
        shape_coeffs = sol.prob.p[3]
        new_shape_coeffs = get_tmp(shape_coeffs, x)
    else
        shape_coeffs = sol.prob.p[4]
        new_shape_coeffs = get_tmp(first(shape_coeffs), x)
    end
    return eval_interpolant!(new_shape_coeffs, prob, x, y, T, sol(t))
end

function DelaunayTriangulation.jump_and_march(prob::FVMProblem, q;
    point_indices=each_point_index(prob),
    m=DelaunayTriangulation.default_num_samples(length(point_indices)),
    try_points=(),
    k=DelaunayTriangulation.select_initial_point(get_triangulation(prob), q; m, point_indices, try_points))
    return jump_and_march(get_triangulation(prob), q; point_indices, m, try_points, k)
end