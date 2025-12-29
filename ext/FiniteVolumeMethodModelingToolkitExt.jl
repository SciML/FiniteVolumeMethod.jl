"""
    FiniteVolumeMethodModelingToolkitExt

Extension module that provides ModelingToolkit/PDESystem integration for FiniteVolumeMethod.jl.

This extension implements the `SciMLBase.discretize` interface, allowing users to define
PDEs symbolically using ModelingToolkit and discretize them using the finite volume method.

# Supported PDE Forms

Currently supports:

  - Diffusion equations: ∂u/∂t = ∇·(D∇u)
  - Reaction-diffusion equations: ∂u/∂t = ∇·(D∇u) + R(u)

where D can be a constant or a function of (x, y).

# Example

```julia
using FiniteVolumeMethod, ModelingToolkit, DelaunayTriangulation

# Define variables
@parameters t x y D
@variables u(..)
Dt = Differential(t)
Dx = Differential(x)
Dy = Differential(y)

# Define the PDE: heat equation
eq = Dt(u(t, x, y)) ~ D * (Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))))

# Create mesh
tri = triangulate_rectangle(0, 1, 0, 1, 20, 20)
mesh = FVMGeometry(tri)

# Define domain and boundary conditions
domains = [t ∈ Interval(0.0, 1.0)]
bcs = [u(t, 0, y) ~ 0, u(t, 1, y) ~ 0, u(t, x, 0) ~ 0, u(t, x, 1) ~ 0]

# Create PDESystem
@named pdesys = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)], [D => 1.0])

# Create discretization and solve
disc = FVMDiscretization(mesh)
prob = discretize(pdesys, disc)
```
"""
module FiniteVolumeMethodModelingToolkitExt

using FiniteVolumeMethod
using FiniteVolumeMethod: FVMGeometry, FVMProblem, SteadyFVMProblem, BoundaryConditions,
                          Neumann, Dirichlet, Dudt, FVMDiscretization
using DelaunayTriangulation
using ModelingToolkit
using ModelingToolkit: PDESystem, get_eqs, get_bcs, get_domain, get_iv, get_dvs, get_ivs,
                       get_ps, get_defaults
using Symbolics
using Symbolics: unwrap, arguments, operation, iscall, Num, substitute
using SciMLBase
using SciMLBase: AbstractDiscretization
using DomainSets
using DomainSets: leftendpoint, rightendpoint

"""
    parse_pde_type(eq, u, t, x, y)

Analyze a PDE equation to determine its type and extract relevant terms.

Returns a NamedTuple with:

  - `type`: :diffusion, :reaction_diffusion, :advection_diffusion, or :general
  - `diffusion_coeff`: The diffusion coefficient (if applicable)
  - `source_term`: The source/reaction term (if applicable)
  - `is_transient`: Whether the equation has a time derivative
"""
function parse_pde_type(eq, u_sym, t_sym, x_sym, y_sym)
    lhs = eq.lhs
    rhs = eq.rhs

    # Check for time derivative on LHS
    is_transient = has_time_derivative(lhs, t_sym)

    # Extract diffusion coefficient and source term from RHS
    diffusion_coeff, source_term = extract_diffusion_and_source(
        rhs, u_sym, t_sym, x_sym, y_sym)

    pde_type = if !isnothing(diffusion_coeff) && isnothing(source_term)
        :diffusion
    elseif !isnothing(diffusion_coeff) && !isnothing(source_term)
        :reaction_diffusion
    else
        :general
    end

    return (type = pde_type, diffusion_coeff = diffusion_coeff, source_term = source_term,
        is_transient = is_transient)
end

"""
    has_time_derivative(expr, t)

Check if expression contains a time derivative.
"""
function has_time_derivative(expr, t_sym)
    expr_unwrapped = unwrap(expr)
    if !iscall(expr_unwrapped)
        return false
    end

    op = operation(expr_unwrapped)
    if op isa Differential && isequal(op.x, t_sym)
        return true
    end

    args = arguments(expr_unwrapped)
    return any(arg -> has_time_derivative(arg, t_sym), args)
end

"""
    extract_diffusion_and_source(expr, u, t, x, y)

Extract diffusion coefficient and source term from PDE right-hand side.

Looks for patterns like:

  - D * (Dxx(u) + Dyy(u)) → diffusion coefficient D
  - D * Laplacian(u) + R(u) → diffusion D, source R
"""
function extract_diffusion_and_source(expr, u_sym, t_sym, x_sym, y_sym)
    diffusion_coeff = nothing
    source_term = nothing

    expr_unwrapped = unwrap(expr)

    # Try to identify Laplacian pattern: ∂²u/∂x² + ∂²u/∂y²
    laplacian_coeff = find_laplacian_coefficient(expr_unwrapped, u_sym, x_sym, y_sym)

    if !isnothing(laplacian_coeff)
        diffusion_coeff = laplacian_coeff
        # Extract remaining terms as source
        remaining = subtract_laplacian_term(
            expr_unwrapped, u_sym, x_sym, y_sym, laplacian_coeff)
        if !is_zero_expr(remaining)
            source_term = remaining
        end
    end

    return diffusion_coeff, source_term
end

"""
    find_laplacian_coefficient(expr, u, x, y)

Find the coefficient of the Laplacian term in an expression.
Returns the coefficient if found, nothing otherwise.
"""
function find_laplacian_coefficient(expr, u_sym, x_sym, y_sym)
    if !iscall(expr)
        return nothing
    end

    op = operation(expr)
    args = arguments(expr)

    # Check for multiplication: coeff * Laplacian or Laplacian * coeff
    if op === (*)
        for (i, arg) in enumerate(args)
            if is_laplacian(arg, u_sym, x_sym, y_sym)
                # Return product of other terms as coefficient
                other_args = [args[j] for j in 1:length(args) if j != i]
                if length(other_args) == 1
                    return other_args[1]
                else
                    return reduce(*, other_args)
                end
            end
        end
    end

    # Check if expr itself is a Laplacian (coefficient = 1)
    if is_laplacian(expr, u_sym, x_sym, y_sym)
        return 1
    end

    # Check for addition: D*Laplacian + source
    if op === (+)
        for arg in args
            coeff = find_laplacian_coefficient(unwrap(arg), u_sym, x_sym, y_sym)
            if !isnothing(coeff)
                return coeff
            end
        end
    end

    return nothing
end

"""
    is_laplacian(expr, u, x, y)

Check if expression is a Laplacian: ∂²u/∂x² + ∂²u/∂y²
"""
function is_laplacian(expr, u_sym, x_sym, y_sym)
    if !iscall(expr)
        return false
    end

    op = operation(expr)

    # Check for sum of second derivatives
    if op === (+)
        args = arguments(expr)
        if length(args) == 2
            return (is_second_derivative(args[1], u_sym, x_sym) &&
                    is_second_derivative(args[2], u_sym, y_sym)) ||
                   (is_second_derivative(args[1], u_sym, y_sym) &&
                    is_second_derivative(args[2], u_sym, x_sym))
        end
    end

    return false
end

"""
    is_second_derivative(expr, u, var)

Check if expression is a second derivative ∂²u/∂var²
"""
function is_second_derivative(expr, u_sym, var_sym)
    expr = unwrap(expr)
    if !iscall(expr)
        return false
    end

    op = operation(expr)

    # Check for D(D(u, var), var) pattern
    if op isa Differential && isequal(op.x, var_sym)
        inner = arguments(expr)[1]
        inner = unwrap(inner)
        if iscall(inner)
            inner_op = operation(inner)
            if inner_op isa Differential && isequal(inner_op.x, var_sym)
                return true
            end
        end
    end

    return false
end

"""
    subtract_laplacian_term(expr, u, x, y, coeff)

Subtract the Laplacian term from an expression to get remaining source terms.
"""
function subtract_laplacian_term(expr, u_sym, x_sym, y_sym, coeff)
    if !iscall(expr)
        return expr
    end

    op = operation(expr)

    if op === (+)
        args = arguments(expr)
        remaining = []
        for arg in args
            if isnothing(find_laplacian_coefficient(unwrap(arg), u_sym, x_sym, y_sym))
                push!(remaining, arg)
            end
        end
        if isempty(remaining)
            return Num(0)
        elseif length(remaining) == 1
            return remaining[1]
        else
            return reduce(+, remaining)
        end
    end

    return Num(0)
end

"""
    is_zero_expr(expr)

Check if an expression is zero.
"""
function is_zero_expr(expr)
    expr = unwrap(expr)
    return expr isa Number && iszero(expr)
end

"""
    build_diffusion_function(D_expr, params, param_vals)

Build a Julia function for the diffusion coefficient from a symbolic expression.
"""
function build_diffusion_function(D_expr, x_sym, y_sym, t_sym, u_sym, params, param_vals)
    D_unwrapped = unwrap(D_expr)

    # If D is just a constant number
    if D_unwrapped isa Number
        D_val = Float64(D_unwrapped)
        return (x, y, t, u, p) -> D_val
    end

    # Create substitution dict for all parameters
    param_dict = Dict{Any, Any}()
    for (i, p) in enumerate(params)
        param_dict[unwrap(p)] = param_vals[i]
        param_dict[p] = param_vals[i]  # Also store Num version
    end

    # Substitute all parameter values
    D_substituted = substitute(D_unwrapped, param_dict)
    D_substituted = unwrap(D_substituted)

    # After substitution, check if it's now a number
    if D_substituted isa Number
        D_val = Float64(D_substituted)
        return (x, y, t, u, p) -> D_val
    end

    # For expressions that still have free variables (x, y, t, u), build a function
    fn_expr = Symbolics.build_function(D_substituted, unwrap(x_sym), unwrap(y_sym),
        unwrap(t_sym), unwrap(u_sym);
        expression = Val{false})

    return (x, y, t, u, p) -> fn_expr(x, y, t, u)
end

"""
    build_source_function(S_expr, x_sym, y_sym, t_sym, u_sym, params, param_vals)

Build a Julia function for the source term from a symbolic expression.
"""
function build_source_function(S_expr, x_sym, y_sym, t_sym, u_sym, params, param_vals)
    S_unwrapped = unwrap(S_expr)

    if S_unwrapped isa Number
        S_val = Float64(S_unwrapped)
        return (x, y, t, u, p) -> S_val
    end

    # Create substitution dict for all parameters
    param_dict = Dict{Any, Any}()
    for (i, p) in enumerate(params)
        param_dict[unwrap(p)] = param_vals[i]
        param_dict[p] = param_vals[i]
    end

    # Substitute parameter values
    S_substituted = substitute(S_unwrapped, param_dict)
    S_substituted = unwrap(S_substituted)

    # After substitution, check if it's now a number
    if S_substituted isa Number
        S_val = Float64(S_substituted)
        return (x, y, t, u, p) -> S_val
    end

    # Build the function
    fn_expr = Symbolics.build_function(S_substituted, unwrap(x_sym), unwrap(y_sym),
        unwrap(t_sym), unwrap(u_sym);
        expression = Val{false})

    return (x, y, t, u, p) -> fn_expr(x, y, t, u)
end

"""
    parse_boundary_condition(bc_eq, u_sym, t_sym, x_sym, y_sym, params, param_vals)

Parse a boundary condition equation to determine its type and function.

Returns:

  - `type`: Dirichlet, Neumann, or Dudt
  - `func`: The boundary condition function (x, y, t, u, p) -> value
"""
function parse_boundary_condition(bc_eq, u_sym, t_sym, x_sym, y_sym, params, param_vals)
    lhs = unwrap(bc_eq.lhs)
    rhs = unwrap(bc_eq.rhs)

    # Check for Dirichlet: u(t, x, y) ~ g(x, y, t)
    if is_dependent_variable(lhs, u_sym)
        bc_type = Dirichlet
        bc_expr = rhs
        # Check for Neumann: ∂u/∂n ~ g or Dx(u) ~ g on boundary
    elseif is_spatial_derivative(lhs, u_sym, x_sym) ||
           is_spatial_derivative(lhs, u_sym, y_sym)
        bc_type = Neumann
        bc_expr = rhs
        # Check for time derivative BC: Dt(u) ~ g
    elseif has_time_derivative(lhs, t_sym)
        bc_type = Dudt
        bc_expr = rhs
    else
        # Default to Dirichlet
        bc_type = Dirichlet
        bc_expr = rhs
    end

    # Create substitution dict for all parameters
    param_dict = Dict{Any, Any}()
    for (i, p) in enumerate(params)
        param_dict[unwrap(p)] = param_vals[i]
        param_dict[p] = param_vals[i]
    end

    # Build the BC function
    bc_expr_substituted = substitute(bc_expr, param_dict)
    bc_expr_substituted = unwrap(bc_expr_substituted)

    if bc_expr_substituted isa Number
        val = Float64(bc_expr_substituted)
        bc_func = (x, y, t, u, p) -> val
    else
        fn_expr = Symbolics.build_function(
            bc_expr_substituted, unwrap(x_sym), unwrap(y_sym),
            unwrap(t_sym); expression = Val{false})
        bc_func = (x, y, t, u, p) -> fn_expr(x, y, t)
    end

    return bc_type, bc_func
end

"""
    is_dependent_variable(expr, u_sym)

Check if expression is the dependent variable u(t, x, y).
"""
function is_dependent_variable(expr, u_sym)
    expr = unwrap(expr)
    if !iscall(expr)
        return false
    end
    op = operation(expr)
    return isequal(op, operation(unwrap(u_sym)))
end

"""
    is_spatial_derivative(expr, u_sym, var_sym)

Check if expression is a spatial derivative ∂u/∂var.
"""
function is_spatial_derivative(expr, u_sym, var_sym)
    expr = unwrap(expr)
    if !iscall(expr)
        return false
    end
    op = operation(expr)
    if op isa Differential && isequal(op.x, var_sym)
        inner = arguments(expr)[1]
        return is_dependent_variable(inner, u_sym)
    end
    return false
end

"""
    extract_initial_condition(pdesys, mesh, u_sym, x_sym, y_sym, params, param_vals)

Extract initial condition values for each mesh node from PDESystem.
"""
function extract_initial_condition(
        pdesys, mesh, u_sym, t_sym, x_sym, y_sym, params, param_vals)
    # Look for initial condition in boundary conditions or defaults
    defaults = get_defaults(pdesys)
    bcs = get_bcs(pdesys)

    # Get number of nodes
    tri = mesh.triangulation
    n_nodes = DelaunayTriangulation.num_points(tri)
    ic = zeros(n_nodes)

    # Create substitution dict for all parameters
    param_dict = Dict{Any, Any}()
    for (i, p) in enumerate(params)
        param_dict[unwrap(p)] = param_vals[i]
        param_dict[p] = param_vals[i]
    end

    # Check for IC in boundary conditions (form: u(0, x, y) ~ f(x, y))
    for bc in bcs
        if is_initial_condition(bc, u_sym, t_sym)
            ic_expr = unwrap(bc.rhs)
            ic_substituted = substitute(ic_expr, param_dict)
            ic_substituted = unwrap(ic_substituted)

            if ic_substituted isa Number
                fill!(ic, Float64(ic_substituted))
            else
                ic_fn = Symbolics.build_function(
                    ic_substituted, unwrap(x_sym), unwrap(y_sym);
                    expression = Val{false})
                for i in 1:n_nodes
                    pt = DelaunayTriangulation.get_point(tri, i)
                    px, py = DelaunayTriangulation.getxy(pt)
                    ic[i] = ic_fn(px, py)
                end
            end
            return ic
        end
    end

    # Default to zero initial condition
    return ic
end

"""
    is_initial_condition(bc, u_sym, t_sym)

Check if a boundary condition is actually an initial condition (at t=0).
"""
function is_initial_condition(bc, u_sym, t_sym)
    lhs = unwrap(bc.lhs)
    if !iscall(lhs)
        return false
    end

    op = operation(lhs)
    if !isequal(op, operation(unwrap(u_sym)))
        return false
    end

    # Check if first argument (time) is 0
    args = arguments(lhs)
    if length(args) >= 1
        t_arg = unwrap(args[1])
        return t_arg isa Number && iszero(t_arg)
    end

    return false
end

"""
    extract_time_span(pdesys, t_sym)

Extract the time span from PDESystem domain.
"""
function extract_time_span(pdesys, t_sym)
    domains = get_domain(pdesys)

    for d in domains
        var = d.variables
        if isequal(unwrap(var), unwrap(t_sym))
            interval = d.domain
            # DomainSets.Interval
            return (leftendpoint(interval), rightendpoint(interval))
        end
    end

    # Default time span
    return (0.0, 1.0)
end

"""
    SciMLBase.discretize(pdesys::PDESystem, disc::FVMDiscretization)

Discretize a PDESystem using the finite volume method.

This function converts a symbolically-defined PDE into an FVMProblem that can be
solved using the standard SciML solve interface.

# Supported PDE Forms

Currently supports PDEs of the form:

  - `∂u/∂t = D∇²u` (diffusion equation)
  - `∂u/∂t = D∇²u + R(x,y,u)` (reaction-diffusion equation)

where D is a constant or parameter.

# Arguments

  - `pdesys::PDESystem`: The ModelingToolkit PDESystem defining the PDE
  - `disc::FVMDiscretization`: The discretization containing the mesh

# Returns

  - `ODEProblem` for transient problems
  - Throws an error for unsupported PDE forms

# Example

```julia
using FiniteVolumeMethod, ModelingToolkit, DelaunayTriangulation

@parameters t x y D
@variables u(..)
Dt = Differential(t)
Dx = Differential(x)
Dy = Differential(y)

eq = Dt(u(t, x, y)) ~ D * (Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))))
bcs = [u(0, x, y) ~ sin(π * x) * sin(π * y)]  # IC
domains = [t ∈ Interval(0.0, 1.0)]

@named pdesys = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)], [D => 1.0])

tri = triangulate_rectangle(0, 1, 0, 1, 20, 20, single_boundary = true)
mesh = FVMGeometry(tri)
disc = FVMDiscretization(mesh)

prob = discretize(pdesys, disc)
# Now solve with: sol = solve(prob, Tsit5())
```
"""
function SciMLBase.discretize(pdesys::PDESystem, disc::FVMDiscretization)
    mesh = disc.mesh

    # Extract variables
    ivs = get_ivs(pdesys)
    dvs = get_dvs(pdesys)
    ps_raw = get_ps(pdesys)
    defaults = get_defaults(pdesys)

    # Identify time and spatial variables
    # Assume first IV is time, next two are spatial
    if length(ivs) < 3
        error("PDESystem must have at least 3 independent variables: time and 2 spatial dimensions")
    end

    t_sym = ivs[1]
    x_sym = ivs[2]
    y_sym = ivs[3]

    # Get the dependent variable
    if length(dvs) != 1
        error("FVMDiscretization currently only supports single-equation PDEs. " *
              "For systems, construct individual FVMProblems and use FVMSystem.")
    end
    u_sym = dvs[1]

    # Get parameter values
    # ps_raw can be:
    # - Vector{Pair{Num, T}} like [D => 1.0] (from @parameters D=1.0)
    # - Vector{Num} like [D] (from @parameters D)
    # We need to extract both the parameter symbols and their default values
    ps = Num[]
    param_vals = Float64[]
    for p in ps_raw
        if p isa Pair
            push!(ps, p.first)
            push!(param_vals, Float64(p.second))
        else
            push!(ps, p)
            # Try to get from defaults, else use 1.0
            val = get(defaults, p, nothing)
            if val === nothing
                val = get(defaults, unwrap(p), 1.0)
            end
            push!(param_vals, Float64(val))
        end
    end

    # Parse the PDE equation
    eqs = get_eqs(pdesys)
    if length(eqs) != 1
        error("FVMDiscretization currently only supports single-equation PDEs.")
    end
    eq = eqs[1]

    pde_info = parse_pde_type(eq, u_sym, t_sym, x_sym, y_sym)

    if pde_info.type == :general
        error("Unsupported PDE form. FVMDiscretization currently supports:\n" *
              "  - Diffusion: ∂u/∂t = D∇²u\n" *
              "  - Reaction-diffusion: ∂u/∂t = D∇²u + R(u)\n" *
              "For general PDEs, please construct FVMProblem directly.")
    end

    # Build diffusion function
    D_expr = pde_info.diffusion_coeff
    diffusion_fn = build_diffusion_function(
        D_expr, x_sym, y_sym, t_sym, u_sym, ps, param_vals)

    # Build source function
    if !isnothing(pde_info.source_term)
        source_fn = build_source_function(pde_info.source_term, x_sym, y_sym, t_sym, u_sym,
            ps, param_vals)
    else
        source_fn = (x, y, t, u, p) -> zero(u)
    end

    # Parse boundary conditions
    bcs = get_bcs(pdesys)

    # Separate BCs from initial conditions
    boundary_bcs = filter(bc -> !is_initial_condition(bc, u_sym, t_sym), bcs)

    # For single-boundary mesh, use first BC or default Neumann
    if isempty(boundary_bcs)
        # Default: zero Neumann (no flux)
        bc_func = (x, y, t, u, p) -> zero(u)
        bc_type = Neumann
    else
        # Use first BC for all boundaries (simplified)
        bc_type, bc_func = parse_boundary_condition(boundary_bcs[1], u_sym, t_sym,
            x_sym, y_sym, ps, param_vals)
    end

    # Create BoundaryConditions
    bc_obj = BoundaryConditions(mesh, bc_func, bc_type)

    # Extract initial condition
    initial_condition = extract_initial_condition(pdesys, mesh, u_sym, t_sym,
        x_sym, y_sym, ps, param_vals)

    # Extract time span
    t_start, t_end = extract_time_span(pdesys, t_sym)

    # Create FVMProblem
    fvm_prob = FVMProblem(mesh, bc_obj;
        diffusion_function = diffusion_fn,
        source_function = source_fn,
        initial_condition = initial_condition,
        initial_time = t_start,
        final_time = t_end)

    # Return ODE problem
    if pde_info.is_transient
        return SciMLBase.ODEProblem(fvm_prob)
    else
        return SciMLBase.SteadyStateProblem(SteadyFVMProblem(fvm_prob))
    end
end

"""
    SciMLBase.symbolic_discretize(pdesys::PDESystem, disc::FVMDiscretization)

Return information about how the PDESystem would be discretized.

For FVM, this returns the FVMProblem directly without converting to an ODEProblem,
which allows inspection of the discretization.
"""
function SciMLBase.symbolic_discretize(pdesys::PDESystem, disc::FVMDiscretization)
    mesh = disc.mesh

    # Extract variables
    ivs = get_ivs(pdesys)
    dvs = get_dvs(pdesys)
    ps_raw = get_ps(pdesys)
    defaults = get_defaults(pdesys)

    if length(ivs) < 3
        error("PDESystem must have at least 3 independent variables: time and 2 spatial dimensions")
    end

    t_sym = ivs[1]
    x_sym = ivs[2]
    y_sym = ivs[3]

    if length(dvs) != 1
        error("FVMDiscretization currently only supports single-equation PDEs.")
    end
    u_sym = dvs[1]

    # Get parameter values (same logic as discretize)
    ps = Num[]
    param_vals = Float64[]
    for p in ps_raw
        if p isa Pair
            push!(ps, p.first)
            push!(param_vals, Float64(p.second))
        else
            push!(ps, p)
            val = get(defaults, p, nothing)
            if val === nothing
                val = get(defaults, unwrap(p), 1.0)
            end
            push!(param_vals, Float64(val))
        end
    end

    eqs = get_eqs(pdesys)
    if length(eqs) != 1
        error("FVMDiscretization currently only supports single-equation PDEs.")
    end
    eq = eqs[1]

    pde_info = parse_pde_type(eq, u_sym, t_sym, x_sym, y_sym)

    if pde_info.type == :general
        error("Unsupported PDE form.")
    end

    D_expr = pde_info.diffusion_coeff
    diffusion_fn = build_diffusion_function(
        D_expr, x_sym, y_sym, t_sym, u_sym, ps, param_vals)

    if !isnothing(pde_info.source_term)
        source_fn = build_source_function(pde_info.source_term, x_sym, y_sym, t_sym, u_sym,
            ps, param_vals)
    else
        source_fn = (x, y, t, u, p) -> zero(u)
    end

    bcs = get_bcs(pdesys)
    boundary_bcs = filter(bc -> !is_initial_condition(bc, u_sym, t_sym), bcs)

    if isempty(boundary_bcs)
        bc_func = (x, y, t, u, p) -> zero(u)
        bc_type = Neumann
    else
        bc_type, bc_func = parse_boundary_condition(boundary_bcs[1], u_sym, t_sym,
            x_sym, y_sym, ps, param_vals)
    end

    bc_obj = BoundaryConditions(mesh, bc_func, bc_type)

    initial_condition = extract_initial_condition(pdesys, mesh, u_sym, t_sym,
        x_sym, y_sym, ps, param_vals)

    t_start, t_end = extract_time_span(pdesys, t_sym)

    fvm_prob = FVMProblem(mesh, bc_obj;
        diffusion_function = diffusion_fn,
        source_function = source_fn,
        initial_condition = initial_condition,
        initial_time = t_start,
        final_time = t_end)

    return fvm_prob
end

end # module
