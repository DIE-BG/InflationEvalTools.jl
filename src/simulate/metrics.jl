"""
    eval_metrics(tray_infl, tray_infl_pob; shortmetrics=false, prefix="") -> Dict

Compute evaluation metrics for inflation trajectories.

This function calculates various statistical metrics to evaluate the performance of simulated inflation trajectories (`tray_infl`) against a parametric population trajectory (`tray_infl_pob`).

# Arguments
- `tray_infl`: Array of simulated inflation trajectories (3D array: periods x 1 x simulations).
- `tray_infl_pob`: Vector or 1-column matrix of the parametric population inflation trajectory.
- `shortmetrics`: Boolean. If `true`, returns a reduced dictionary with key metrics (MSE, RMSE, MAE, ME, AbsME, Huber, Correlation) and their standard errors. Default is `true`.
- `prefix`: String. Optional prefix for the keys in the output dictionary.

# Returns
A dictionary containing the computed metrics.

If `shortmetrics=true`, the dictionary includes:
- `mse`, `mse_std_error`: Mean Squared Error and its standard error.
- `rmse`, `rmse_std_error`: Root Mean Squared Error and its standard error.
- `mae`, `mae_std_error`: Mean Absolute Error and its standard error.
- `me`, `me_std_error`: Mean Error (Bias) and its standard error.
- `absme`, `absme_std_error`: Absolute Mean Error and its standard error.
- `huber`, `huber_std_error`: Huber Loss and its standard error.
- `corr`: Average correlation between simulations and population trajectory.

If `shortmetrics=false`, the dictionary additionally includes:
- `std_mse_dist`: Standard deviation of the MSE distribution.
- `std_sqerr_dist`: Standard deviation of the squared error distribution.
- `mse_bias`: Squared bias component of MSE decomposition.
- `mse_var`: Variance component of MSE decomposition.
- `mse_cov`: Covariance component of MSE decomposition.
- `T`: Number of periods.
- `B`: Number of simulations.
"""
function eval_metrics(tray_infl::AbstractArray{F, 3}, tray_infl_pob::AbstractArray{F, 1}; shortmetrics = true, prefix = "") where {F <: AbstractFloat}
    _prefix = prefix == "" ? "" : prefix * "_"
    T = size(tray_infl, 1)
    K = size(tray_infl, 3)

    # Error distributions
    #err_dist = compute_error_distribution(tray_infl, tray_infl_pob)
    err_dist = tray_infl .- tray_infl_pob

    # MSE
    mse = mean(x -> x^2, err_dist)

    # Squared error distribution (average over time for each simulation)
    mse_dist = vec(mean(x -> x^2, err_dist, dims = 1))
    # Standard deviation of the MSE distribution for the full period
    std_mse_dist = std(mse_dist, mean = mse)
    # Standard error of simulation of the obtained average value
    mse_std_error = std_mse_dist / sqrt(K)

    # RMSE, MAE, ME
    rmse = mean(sqrt, mse_dist)
    mae = mean(abs, err_dist)
    me = mean(err_dist)
    absme = abs(me)

    # Huber loss ~ combines the properties of MSE and MAE
    huber = mean(huber_loss, err_dist)

    # Correlation
    #corr_dist = compute_corr(tray_infl, tray_infl_pob)
    corr_dist = first.(cor.(eachslice(tray_infl, dims = 3), Ref(tray_infl_pob)))
    corr = mean(corr_dist)

    # Return short metrics
    if shortmetrics
        return Dict(
            Symbol(_prefix, "mse") => mse,
            Symbol(_prefix, "rmse") => rmse,
            Symbol(_prefix, "mae") => mae,
            Symbol(_prefix, "me") => me,
            Symbol(_prefix, "absme") => absme,
            Symbol(_prefix, "huber") => huber,
            Symbol(_prefix, "corr") => corr,
        )
    end

    # Standard errors
    rmse_std_error = std(sqrt.(mse_dist), mean = rmse) / sqrt(K)
    mae_std_error = std(mean(abs, err_dist, dims = 2), mean = mae) / sqrt(K)
    me_std_error = std(mean(err_dist, dims = 2), mean = me) / sqrt(K)
    huber_std_error = std(mean(huber_loss, err_dist, dims = 2), mean = huber) / sqrt(T * K)

    # Standard deviation of the squared error ~ all periods and realizations
    sq_err_dist = err_dist .^ 2
    std_sqerr_dist = std(sq_err_dist, mean = mse)

    ## Additive decomposition of the MSE

    # Bias^2
    me_dist = vec(mean(err_dist, dims = 1))
    mse_bias = mean(x -> x^2, me_dist)

    # Variance component
    s_param = std(tray_infl_pob, corrected = false)
    s_tray_infl = vec(std(tray_infl, dims = 1, corrected = false))
    mse_var = mean(s -> (s - s_param)^2, s_tray_infl)

    # Correlation component
    mse_cov_dist = @. 2 * (1 - corr_dist) * s_param * s_tray_infl
    mse_cov = mean(mse_cov_dist)

    # Dictionary of metrics to return
    return Dict(
        Symbol(_prefix, "mse") => mse,
        Symbol(_prefix, "mse_std_error") => mse_std_error,
        Symbol(_prefix, "std_mse_dist") => std_mse_dist,
        Symbol(_prefix, "std_sqerr_dist") => std_sqerr_dist,
        Symbol(_prefix, "rmse") => rmse,
        Symbol(_prefix, "rmse_std_error") => rmse_std_error,
        Symbol(_prefix, "mae") => mae,
        Symbol(_prefix, "mae_std_error") => mae_std_error,
        Symbol(_prefix, "me") => me,
        Symbol(_prefix, "me_std_error") => me_std_error,
        Symbol(_prefix, "absme") => absme,
        Symbol(_prefix, "absme_std_error") => me_std_error,
        Symbol(_prefix, "corr") => corr,
        Symbol(_prefix, "huber") => huber,
        Symbol(_prefix, "huber_std_error") => huber_std_error,
        Symbol(_prefix, "mse_bias") => mse_bias,
        Symbol(_prefix, "mse_var") => mse_var,
        Symbol(_prefix, "mse_cov") => mse_cov,
        Symbol(_prefix, "T") => T,
        Symbol(_prefix, "B") => K
    )
end

function eval_metrics(tray_infl::AbstractArray{F, 3}, tray_infl_pob::AbstractArray{F, 3}; shortmetrics = true, prefix = "") where {F <: AbstractFloat}
    T_infl, M_infl, K_infl = size(tray_infl)
    T_param, M_param, N_batches = size(tray_infl_pob)

    @assert K_infl % N_batches == 0 "The number of batches inferred from the simulations in `traj_infl` and `traj_infl_param` should be an integer."

    # number of simulations per batch
    N_sim = Int(K_infl / N_batches)

    results_b = Vector{Dict{Symbol, F}}(undef, N_batches)

    for batch in 1:N_batches
        lower_limit = 1 + N_sim * (batch - 1)
        upper_limit = N_sim * batch
        tray_infl_batch = @view tray_infl[:, :, lower_limit:upper_limit]
        tray_infl_pob_batch = @view tray_infl_pob[:, :, batch]
        results_b[batch] = eval_metrics(
            tray_infl_batch, vec(tray_infl_pob_batch);
            shortmetrics = shortmetrics, prefix = prefix
        )
    end

    results = Dict{Symbol, F}()
    for key in keys(results_b[1])
        results[key] = mean([results_b[b][key] for b in 1:N_batches])
    end

    return results
end

# Huber loss
function huber_loss(x::Real; a = 1)
    if abs(x) <= a
        return x^2 / 2
    else
        return a * (abs(x) - a / 2)
    end
end


# Mean squared error - MSE: average across time and number of
# realizations of the errors between the inflation trajectories tray_infl and
# the parametric inflation trajectory tray_pob
function _mse(tray_infl, tray_infl_pob)
    return mean(x -> x^2, tray_infl .- tray_infl_pob)
end


# Metrics for linear combinations of estimators
"""
    combination_metrics(tray_infl, tray_infl_param, w; kwargs...) 

Metrics for linear combination measures. The trajectories in
`tray_infl` are combined with the weights `w` and the evaluation metrics are computed
using the parametric trajectory `tray_infl_param`. 

Optional arguments (`kwargs...`) are passed to the function
[`eval_metrics`](@ref).
"""
function combination_metrics(tray_infl, tray_infl_param, w; kwargs...)
    tray_infl_comb = sum(tray_infl .* w', dims = 2)
    return eval_metrics(tray_infl_comb, tray_infl_param; kwargs...)
end
