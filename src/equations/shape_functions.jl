# primitive: get the shape function coefficients for a non-system
@inline function get_shape_function_coefficients(props::TriangleProperties, T, u, ::AbstractFVMProblem)
    i, j, k = indices(T)
    s₁, s₂, s₃, s₄, s₅, s₆, s₇, s₈, s₉ = props.shape_function_coefficients
    α = s₁ * u[i] + s₂ * u[j] + s₃ * u[k]
    β = s₄ * u[i] + s₅ * u[j] + s₆ * u[k]
    γ = s₇ * u[i] + s₈ * u[j] + s₉ * u[k]
    return α, β, γ
end

# primitive: get the shape function coefficients for a system
@inline function get_shape_function_coefficients(props::TriangleProperties, T, u, prob::FVMSystem) 
    i, j, k = indices(T)
    s₁, s₂, s₃, s₄, s₅, s₆, s₇, s₈, s₉ = props.shape_function_coefficients
    α = ntuple(ℓ -> s₁ * u[ℓ, i] + s₂ * u[ℓ, j] + s₃ * u[ℓ, k], _neqs(prob))
    β = ntuple(ℓ -> s₄ * u[ℓ, i] + s₅ * u[ℓ, j] + s₆ * u[ℓ, k], _neqs(prob))
    γ = ntuple(ℓ -> s₇ * u[ℓ, i] + s₈ * u[ℓ, j] + s₉ * u[ℓ, k], _neqs(prob))
    return α, β, γ
end