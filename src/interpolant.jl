"""
    eval_interpolant!(αβγ, prob::FVMProblem, x, y, T, u)
    eval_interpolant(sol, x, y, t_idx::Integer, T)
    eval_interpolant(sol, x, y, t::Number, T)

Evaluates the interpolant corresponding to the `FVMProblem` at the point `(x, y)` and time `t`
(or `sol.t[t_idx]`). 
"""
function eval_interpolant end
@inline function eval_interpolant!(αβγ, prob::FVMProblem, x, y, T, u)
    if T ∉ get_elements(prob)
        T = DelaunayTriangulation.shift_triangle_1(T)
    end
    if T ∉ get_elements(prob)
        T = DelaunayTriangulation.shift_triangle_1(T)
    end
    linear_shape_function_coefficients!(αβγ, u, prob, T)
    α = getα(αβγ)
    β = getβ(αβγ)
    γ = getγ(αβγ)
    return α * x + β * y + γ
end
@inline function eval_interpolant(sol, x, y, t_idx::Integer, T)
    prob = sol.prob.p[1]
    shape_coeffs = sol.prob.p[3]
    new_shape_coeffs = get_tmp(shape_coeffs, x)
    return eval_interpolant!(new_shape_coeffs, prob, x, y, T, sol.u[t_idx])
end
@inline function eval_interpolant(sol, x, y, t::Number, T)
    prob = sol.prob.p[1]
    shape_coeffs = sol.prob.p[3]
    new_shape_coeffs = get_tmp(shape_coeffs, x)
    return eval_interpolant!(new_shape_coeffs, prob, x, y, T, sol(t))
end

function DelaunayTriangulation.jump_and_march(x, y, prob::FVMProblem;
    pt_idx=DelaunayTriangulation._eachindex(get_points(prob)),
    m=ceil(Int64, length(pt_idx)^(1 / 3)),
    try_points=(),
    k=DelaunayTriangulation.select_initial_point(get_points(prob), (x, y); m, pt_idx, try_points),
    TriangleType::Type{V}=get_element_type(prob)) where {V}
    adj = get_adjacent(prob)
    adj2v = get_adjacent2vertex(prob)
    DG = get_neighbours(prob)
    pts = get_points(prob)
    q = (x, y)
    return jump_and_march(q, adj, adj2v, DG, pts; pt_idx, m, k, TriangleType)
end