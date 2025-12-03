## Analytical solution method for optimal MSE linear combination weights

"""
    combination_weights(tray_infl, tray_infl_param) -> Vector{<:AbstractFloat}

Obtains the optimal weights from the analytical solution to the problem of
minimizing the mean squared error of the linear combination of inflation
estimators in `tray_infl` using the parametric inflation trajectory
`tray_infl_param`. 

Returns a vector with the weights associated with each estimator in the
columns of `tray_infl`.

See also: [`ridge_combination_weights`](@ref),
[`lasso_combination_weights`](@ref), [`share_combination_weights`](@ref),
[`elastic_combination_weights`](@ref). 
"""
function combination_weights(tray_infl, tray_infl_param)
    # Get the weights matrix XᵀX and vector Xᵀπ
    XᵀX, Xᵀπ = average_mats(tray_infl, tray_infl_param)
    @debug "Determinant of the coefficient matrix" det(XᵀX)

    # Optimal least squares combination weights
    a_optim = XᵀX \ Xᵀπ
    return a_optim
end

# Design and covariance matrix with parameter, averaged over time and
# across realizations
"""
    average_mats(tray_infl, tray_infl_param) -> (Matrix{<:AbstractFloat}, Vector{<:AbstractFloat})

Obtains the matrices `XᵀX` and `Xᵀπ` for the mean squared error minimization
problem. 
"""
function average_mats(tray_infl, tray_infl_param)
    # Number of weights, observations, and simulations
    T, n, K = size(tray_infl)

    # Build the coefficient matrix
    XᵀX = zeros(eltype(tray_infl), n, n)
    XᵀX_temp = zeros(eltype(tray_infl), n, n)
    for j in 1:K
        tray = @view tray_infl[:, :, j]
        LinearAlgebra.mul!(XᵀX_temp, tray', tray)
        XᵀX_temp ./= T
        XᵀX += XᵀX_temp
    end
    # Average over number of realizations
    XᵀX /= K

    # Intercepts as a function of the parameter and the trajectories to combine
    Xᵀπ = zeros(eltype(tray_infl), n)
    Xᵀπ_temp = zeros(eltype(tray_infl), n)
    for j in 1:K
        tray = @view tray_infl[:, :, j]
        LinearAlgebra.mul!(Xᵀπ_temp, tray', tray_infl_param)
        Xᵀπ_temp ./= T
        Xᵀπ += Xᵀπ_temp
    end
    # Average over number of realizations
    Xᵀπ /= K

    return XᵀX, Xᵀπ
end

# Ridge combination weights with regularization parameter lambda
"""
    ridge_combination_weights(tray_infl, tray_infl_param, lambda; 
        penalize_all = true) -> Vector{<:AbstractFloat}

Obtains optimal Ridge weights through the analytical solution to the problem of
minimizing the mean squared error of the linear combination of inflation
estimators in `tray_infl` using the parametric inflation trajectory
`tray_infl_param`, regularized with the L2 norm of the weights, weighted by the
parameter `lambda`.

Returns a vector with the weights associated with each estimator in the
columns of `tray_infl`.

Optional parameters:  
- `penalize_all` (`Bool`): indicates whether to apply regularization to all
  weights. If false, regularization is applied from the second to the last
  component of the weights vector. Default is `true`.

See also: [`combination_weights`](@ref), [`lasso_combination_weights`](@ref),
[`share_combination_weights`](@ref), [`elastic_combination_weights`](@ref).
"""
function ridge_combination_weights(
        tray_infl::AbstractArray{F, 3}, tray_infl_param, lambda;
        penalize_all = true
    ) where {F}

    # If lambda == 0, least squares solution
    lambda == 0 && return combination_weights(tray_infl, tray_infl_param)

    # Get the weights matrix XᵀX and vector Xᵀπ
    XᵀX, Xᵀπ = average_mats(tray_infl, tray_infl_param)
    λ = convert(F, lambda)
    n = size(tray_infl, 2)

    # Optimal Ridge combination weights
    Iₙ = I(n)
    # If penalize_all=false, do not penalize the first component, which should
    # correspond to the regression intercept. For this, the first column of
    # tray_infl should contain 1's.
    if !penalize_all
        Iₙ[1] = 0
    end

    XᵀX′ = XᵀX + λ * Iₙ
    @debug "Determinant of the coefficient matrix" det(XᵀX) det(XᵀX′)
    a_ridge = XᵀX′ \ Xᵀπ
    return a_ridge
end


# Ponderadores de combinación lasso con parámetro de regularización lambda
"""
    lasso_combination_weights(tray_infl, tray_infl_param, lambda; 
        max_iterations::Int = 1000, 
        alpha = F(0.005), 
        tol = F(1e-4), 
        show_status = true, 
        return_cost = false, 
        penalize_all = true) -> Vector{<:AbstractFloat}

Obtiene ponderadores óptimos de LASSO a través de una aproximación iterativa al
problema de minimización del error cuadrático medio de la combinación lineal de
estimadores de inflación en `tray_infl` utilizando la trayectoria de inflación
paramétrica `tray_infl_param`, regularizada con la norma L1 de los ponderadores,
ponderada por el parámetro `lambda`.

Los parámetros opcionales son: 
- `max_iterations::Int = 1000`: número máximo de iteraciones. 
- `alpha::AbstractFloat = 0.001`: parámetro de aproximación o avance del
  algoritmo de gradiente próximo. 
- `tol::AbstractFloat = 1e-4`: desviación absoluta de la función de costo. Si la
  función de costo varía en términos absolutos menos que `tol` de una iteración
  a otra, el algoritmo de gradiente se detiene. 
- `show_status::Bool = true`: mostrar estado del algoritmo iterativo.
- `return_cost::Bool = false`: indica si devuelve el vector de historia de la
  función de costo de entrenamiento. 
- `penalize_all::Bool = true`: indica si aplicar la regularización a todos los
  ponderadores. Si es falso, se aplica la regularización a partir del segundo al
  último componente del vector de ponderaciones.

Devuelve un vector con los ponderadores asociados a cada estimador en las
columnas de `tray_infl`.

Ver también: [`combination_weights`](@ref), [`ridge_combination_weights`](@ref),
[`share_combination_weights`](@ref), [`elastic_combination_weights`](@ref).
"""
function lasso_combination_weights(
        tray_infl::AbstractArray{F, 3}, tray_infl_param, lambda;
        max_iterations::Int = 1000,
        alpha = F(0.001),
        tol = F(1.0e-4),
        show_status = true,
        return_cost = false,
        penalize_all = true
    ) where {F}

    # Si lambda == 0, solución de mínimos cuadrados
    lambda == 0 && return combination_weights(tray_infl, tray_infl_param)

    T, n, _ = size(tray_infl)

    λ = convert(F, lambda)
    α = convert(F, alpha)
    β = zeros(F, n)
    cost_vals = zeros(F, max_iterations)
    XᵀX, Xᵀπ = average_mats(tray_infl, tray_infl_param)
    πᵀπ = mean(x -> x^2, tray_infl_param)

    if show_status
        println("Optimización iterativa para LASSO:")
        println("----------------------------------")
    end

    # Proximal gradient descent
    for t in 1:max_iterations
        # Computar el gradiente respecto de β
        grad = (XᵀX * β) - Xᵀπ

        # Proximal gradient
        β = proxl1norm(β - α * grad, α * λ; penalize_all)

        # Métrica de costo = MSE + λΣᵢ|βᵢ|
        mse = β' * XᵀX * β - 2 * β' * Xᵀπ + πᵀπ
        l1cost = penalize_all ? sum(abs, β) : sum(abs, (@view β[2:end]))
        cost_vals[t] = mse + λ * l1cost
        abstol = t > 1 ? abs(cost_vals[t] - cost_vals[t - 1]) : 100.0f0

        if show_status && t % 100 == 0
            println("Iter: ", t, " cost = ", cost_vals[t], "  |Δcost| = ", abstol)
        end

        abstol < tol && break
    end

    return_cost && return β, cost_vals
    return β
end

# Operador próximo para la norma L1 del vector z
function proxl1norm(z, α; penalize_all = true)
    proxl1 = z - clamp.(z, Ref(-α), Ref(α))

    # penalize_all = false : no penalizar del intercepto
    if !penalize_all
        proxl1[1] = z[1]
    end

    return proxl1
end


## Ponderadores de combinación restringidos
# Se restringe el problema de optimización para que la suma de los ponderadores
# sea igual a 1 y que todas las ponderaciones sean no negativas.
"""
    function share_combination_weights(tray_infl::AbstractArray{F, 3}, tray_infl_param; 
        restrict_all::Bool = true, 
        show_status::Bool = false) where F -> Vector{F}

Obtiene ponderadores no negativos, cuya suma es igual a 1. Estos ponderadores se
pueden interpretar como participaciones en la combinación lineal. 

Los parámetros opcionales son: 
- `show_status::Bool = false`: mostrar estado del proceso de optimización con
  Ipopt y JuMP. 
- `restrict_all::Bool = true`: indica si aplicar la restricción de la suma de
  ponderadores a todas las entradas del vector de ponderaciones. Si es `false`,
  se aplica la restricción a partir de la segunda entrada. Esto es para que si
  el primer ponderador corresponde a un término constante, este no sea
  restringido. 
"""
function share_combination_weights(
        tray_infl::AbstractArray{F, 3}, tray_infl_param;
        restrict_all::Bool = true,
        show_status::Bool = false
    ) where {F}

    # Insumos para la función de pérdida cuadrática
    n = size(tray_infl, 2)
    XᵀX, Xᵀπ = average_mats(tray_infl, tray_infl_param)
    πᵀπ = mean(x -> x^2, tray_infl_param)

    # Si restrict_all == false, se restringe la suma de ponderadores igual a 1 a
    # partir de la segunda posición del vector de ponderadores β
    r = restrict_all ? 1 : 2

    # Problema de optimización restringida
    model = Model(Ipopt.Optimizer)
    @variable(model, β[1:n] >= 0)
    @constraint(model, sum(β[r:n]) == 1)
    @objective(model, Min, β' * XᵀX * β - 2 * β' * Xᵀπ + πᵀπ)

    # Obtener la solución numérica
    show_status || set_silent(model)
    optimize!(model)
    return convert.(F, JuMP.value.(β))
end


# Elastic net
"""
    elastic_combination_weights(tray_infl, tray_infl_param, lambda, gamma; 
        max_iterations::Int = 1000, 
        alpha = 0.001, 
        tol = 1e-4, 
        show_status = true, 
        return_cost = false, 
        penalize_all = true) -> Vector{<:AbstractFloat}

Obtiene ponderadores óptimos de [Elastic
Net](https://en.wikipedia.org/wiki/Elastic_net_regularization) a través de una
aproximación iterativa al problema de minimización del error cuadrático medio de
la combinación lineal de estimadores de inflación en `tray_infl` utilizando la
trayectoria de inflación paramétrica `tray_infl_param`, regularizada con la
norma L1 y L2 de los ponderadores, ponderada por el parámetro `lambda`. El
porcentaje de regularización de la norma L1 se controla con el parámetro
`gamma`.

Los parámetros opcionales son: 
- `max_iterations::Int = 1000`: número máximo de iteraciones. 
- `alpha::AbstractFloat = 0.001`: parámetro de aproximación o avance del
  algoritmo de gradiente próximo. 
- `tol::AbstractFloat = 1e-4`: desviación absoluta de la función de costo. Si la
  función de costo varía en términos absolutos menos que `tol` de una iteración
  a otra, el algoritmo de gradiente se detiene. 
- `show_status::Bool = true`: mostrar estado del algoritmo iterativo.
- `return_cost::Bool = false`: indica si devuelve el vector de historia de la
  función de costo de entrenamiento. 
- `penalize_all::Bool = true`: indica si aplicar la regularización a todos los
  ponderadores. Si es falso, se aplica la regularización a partir del segundo al
  último componente del vector de ponderaciones.

Devuelve un vector con los ponderadores asociados a cada estimador en las
columnas de `tray_infl`.

Ver también: [`combination_weights`](@ref), [`ridge_combination_weights`](@ref),
[`share_combination_weights`](@ref), [`lasso_combination_weights`](@ref).
"""
function elastic_combination_weights(
        tray_infl::AbstractArray{F, 3}, tray_infl_param, lambda, gamma;
        max_iterations::Int = 1000,
        alpha = F(0.001),
        tol = F(1.0e-4),
        show_status::Bool = true,
        return_cost::Bool = false,
        penalize_all::Bool = true
    ) where {F}

    # Si lambda == 0, solución de mínimos cuadrados
    lambda == 0 && return combination_weights(tray_infl, tray_infl_param)

    n = size(tray_infl, 2)

    λ = convert(F, lambda)
    γ = convert(F, gamma)
    α = convert(F, alpha)
    β = zeros(F, n)
    cost_vals = zeros(F, max_iterations)
    XᵀX, Xᵀπ = average_mats(tray_infl, tray_infl_param)
    πᵀπ = mean(x -> x^2, tray_infl_param)

    if show_status
        println("Optimización iterativa para Elastic Net:")
        println("----------------------------------------")
    end

    # Proximal gradient descent
    for t in 1:max_iterations
        # Computar el gradiente respecto de β
        grad = (XᵀX * β) - Xᵀπ + λ * (1 - γ) * β

        # Proximal gradient
        β = proxl1norm(β - α * grad, α * λ * γ; penalize_all)

        # Métrica de costo = 0.5MSE + 0.5λ(1-γ)Σᵢ||βᵢ||^2 + γλΣᵢ|βᵢ|
        mse = β' * XᵀX * β - 2 * β' * Xᵀπ + πᵀπ
        l1cost = penalize_all ? sum(abs, β) : sum(abs, (@view β[2:end]))
        l2cost = penalize_all ? sum(x -> x^2, β) : sum(x -> x^2, (@view β[2:end]))
        cost_vals[t] = (1 // 2)mse + λ * γ * l1cost + (1 // 2)λ * (1 - γ) * l2cost
        abstol = t > 1 ? abs(cost_vals[t] - cost_vals[t - 1]) : 100.0f0

        if show_status && t % 100 == 0
            println("Iter: ", t, " cost = ", cost_vals[t], "  |Δcost| = ", abstol)
        end

        abstol < tol && break
    end

    return_cost && return β, cost_vals
    return β
end


"""
    share_combination_weights_rmse(tray_infl::AbstractArray{F,3}, tray_infl_param::AbstractArray{F,3}) -> Vector{F}

Compute a vector of non-negative weights `β` that sum to 1 by minimizing a nonlinear loss function based on the average root mean squared error (RMSE).
"""
function share_combination_weights_rmse(traj_infl::AbstractArray{F, 3}, traj_infl_param::AbstractArray{F, 3}) where {F}

    T_traj, N_traj, K_traj = size(traj_infl)
    T_param, N_param, K_param = size(traj_infl_param)

    @assert T_traj == T_param "The trajectories and the parameter should have the same number of periods."
    @assert K_traj % K_param == 0 "The number of trajectories should be the number of simulations times the numbers of batches."

    # number of simulations per batch
    N_sim = convert(Int, K_traj / K_param)

    function rmse_loss(β)
        # The first level of the loop represents the batch in the trend simulation
        rmse_b = Vector(undef, K_param)
        for b in 1:K_param

            param_b = @view traj_infl_param[:, :, b]
            traj_b = @view traj_infl[:, :, (1 + N_sim * (b - 1)):(N_sim * b)]

            # The second level of the loop represents the simulations per batch
            rmse_k = Vector(undef, N_sim)
            for k in 1:N_sim
                # The third level is the accumulation of the squared error
                ∑e² = F(0)
                for t in 1:T_traj
                    ∑e² += (traj_b[t, :, k]' * β - param_b[t])^2
                end
                rmse_k[k] = sqrt((1 / T_traj) * ∑e²)
            end

            rmse_b[b] = mean(rmse_k)
        end

        return mean(rmse_b)
    end

    # restricted optimization problem
    model = Model(Ipopt.Optimizer)
    @variable(model, β[1:N_traj] >= 0)
    @constraint(model, sum(β[1:N_traj]) == 1)

    @objective(model, Min, rmse_loss(β))

    optimize!(model)
    return convert.(F, JuMP.value.(β))
end


"""
    share_combination_weights_absme(tray_infl::AbstractArray{F,3}, tray_infl_param) -> Vector{F}

Compute a vector of non-negative weights `β` that sum to 1 by minimizing a
nonlinear loss function based on the average absolute mean error (ABSME).

The objective function is:

    ABSME(β) = (1/K) ⋅ Σₖ |(1/T) ⋅ Σₜ (ĥπₜₖ(β) − πₜ|

where:

- ĥπₜₖ(β) = tray_infl[t, :, k]' * β  is the predicted linear combination,
- πₜ is the target series,
- T is the number of time periods,
- K is the number of trajectories or models.

The resulting vector `β` represents the shares in the optimal linear combination
under the ABSME criterion.
"""
function share_combination_weights_absme(tray_infl::AbstractArray{F, 3}, tray_infl_param) where {F}

    T, N, K = size(tray_infl)

    @assert T == length(tray_infl_param) "The trajectories and the parameter should have the same number of periods."

    function absme_loss(β)
        absme_k = Vector(undef, K)
        for k in 1:K
            ∑e = F(0)
            for t in 1:T
                ∑e += (tray_infl[t, :, k]' * β - tray_infl_param[t])
            end
            absme_k[k] = abs((1 / T) * ∑e)
        end
        return mean(absme_k)
    end

    # Problema de optimización restringida
    model = Model(Ipopt.Optimizer)
    @variable(model, β[1:N] >= 0)
    @constraint(model, sum(β[1:N]) == 1)

    @objective(model, Min, absme_loss(β))

    optimize!(model)
    return convert.(F, JuMP.value.(β))
end


"""
    share_combination_weights_absme(tray_infl::AbstractArray{F,3}, tray_infl_param) -> Vector{F}

Compute a vector of non-negative weights `β` that sum to 1 by minimizing a
nonlinear loss function based on the average absolute mean error (ABSME).

The objective function is:

    ABSME(β) = (1/K) ⋅ Σₖ |(1/T) ⋅ Σₜ (ĥπₜₖ(β) − πₜ|

where:

- ĥπₜₖ(β) = tray_infl[t, :, k]' * β  is the predicted linear combination,
- πₜ is the target series,
- T is the number of time periods,
- K is the number of trajectories or models.

The resulting vector `β` represents the shares in the optimal linear combination
under the ABSME criterion.
"""
function share_combination_weights_corr(tray_infl::AbstractArray{F, 3}, tray_infl_param) where {F}

    T, N, K = size(tray_infl)

    @assert T == length(tray_infl_param) "The trajectories and the parameter should have the same number of periods."

    function corr_loss(β)
        corr_k = Vector(undef, K)
        for k in 1:K
            corr_k[k] = cor(tray_infl[:, :, k] * β, tray_infl_param)
        end
        return mean(corr_k)
    end

    # Problema de optimización restringida
    model = Model(Ipopt.Optimizer)
    @variable(model, β[1:N] >= 0)
    @constraint(model, sum(β[1:N]) == 1)

    @objective(model, Max, corr_loss(β))

    optimize!(model)
    return convert.(F, JuMP.value.(β))
end
