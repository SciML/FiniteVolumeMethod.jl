# get the relevant quantities for a control volume edge not on the boundary 
"""
    get_cv_components(props, edge_index)

Get the quantities for a control volume edge interior to the associated triangulation,
relative to the `edge_index`th edge of the triangle corresponding to `props`.

# Outputs 
- `x`: The `x`-coordinate of the edge's midpoint. 
- `y`: The `y`-coordinate of the edge's midpoint.
- `nx`: The `x`-component of the edge's normal vector.
- `ny`: The `y`-component of the edge's normal vector.
- `ℓ`: The length of the edge.
"""
@inline function get_cv_components(props, edge_index)
    x, y = props.cv_edge_midpoints[edge_index]
    nx, ny = props.cv_edge_normals[edge_index]
    ℓ = props.cv_edge_lengths[edge_index]
    return x, y, nx, ny, ℓ
end

# get the relevant quantities for a control volume edge on a boundary edge
"""
    get_boundary_cv_components(tri::Triangulation, i, j)

Get the quantities for both control volume edges lying a boundary edge `(i, j)`.

# Outputs 
- `nx`: The `x`-component of the edge's normal vector.
- `ny`: The `y`-component of the edge's normal vector.
- `mᵢx`: The `x`-coordinate of the midpoint of the `i`th vertex and the edge's midpoint.
- `mᵢy`: The `y`-coordinate of the midpoint of the `i`th vertex and the edge's midpoint.
- `mⱼx`: The `x`-coordinate of the midpoint of the `j`th vertex and the edge's midpoint.
- `mⱼy`: The `y`-coordinate of the midpoint of the `j`th vertex and the edge's midpoint.
- `ℓᵢ`: Half the length of the boundary edge, which is the length of the control volume edge.
- `T`: The triangle containing the boundary edge.
- `props`: The [`TriangleProperties`](@ref) for `T`.
"""
@inline function get_boundary_cv_components(mesh, i, j)
    p, q = get_point(mesh, i, j)
    px, py = getxy(p)
    qx, qy = getxy(q)
    ℓᵢⱼ = norm((qx - px, qy - py))
    nx, ny = (qy - py) / ℓᵢⱼ, -(qx - px) / ℓᵢⱼ
    mᵢⱼx, mᵢⱼy = (px + qx) / 2, (py + qy) / 2
    mᵢx, mᵢy = (px + mᵢⱼx) / 2, (py + mᵢⱼy) / 2
    mⱼx, mⱼy = (qx + mᵢⱼx) / 2, (qy + mᵢⱼy) / 2
    ℓᵢ = norm((mᵢⱼx - px, mᵢⱼy - py))
    # ℓⱼ = norm((mᵢⱼx - qx, mᵢⱼy - qy)) # same as the above
    k = get_adjacent(mesh.triangulation, i, j)
    T = (i, j, k)
    T, props = _safe_get_triangle_props(mesh, T)
    return nx, ny, mᵢx, mᵢy, mⱼx, mⱼy, ℓᵢ, T, props
end

