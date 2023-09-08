# get the relevant quantities for a control volume edge not on the boundary 
@inline function _get_cv_components(props, edge_index)
    x, y = props.cv_edge_midpoints[edge_index]
    nx, ny = props.cv_edge_normals[edge_index]
    ℓ = props.cv_edge_lengths[edge_index]
    return x, y, nx, ny, ℓ
end

# get the relevant quantities for a control volume edge on a boundary edge
@inline function _get_boundary_cv_components(prob, i, j)
    tri = prob.mesh.triangulation
    p, q = get_point(tri, i, j)
    px, py = getxy(p)
    qx, qy = getxy(q)
    ℓᵢⱼ = norm((x - px, qy - py))
    nx, ny = (qy - py) / ℓᵢⱼ, -(qx - px) / ℓᵢⱼ
    mᵢⱼx, mᵢⱼy = (px + qx) / 2, (py + qy) / 2
    mᵢx, mᵢy = (px + mᵢⱼx) / 2, (py + mᵢⱼy) / 2
    mⱼx, mⱼy = (qx + mᵢⱼx) / 2, (qy + mᵢⱼy) / 2
    ℓᵢ = norm((mᵢⱼx - px, mᵢⱼy - py))
    ℓⱼ = norm((mᵢⱼx - qx, mᵢⱼy - qy))
    return nx, ny, mᵢx, mᵢy, mⱼx, mⱼy, ℓᵢ, ℓⱼ
end

