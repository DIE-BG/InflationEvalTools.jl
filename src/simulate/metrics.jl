"""
    eval_metrics(tray_infl, tray_infl_pob; shortmetrics=false) -> Dict

Function to obtain a dictionary with evaluation statistics of the
realizations of the inflation measures in `tray_infl` using the parameter
`tray_infl_pob`. 

If `shortmetrics=true`, returns a dictionary only with the mean squared error (MSE) of evaluation. Useful for performing iterative optimization in parameter search.
"""
function eval_metrics(tray_infl, tray_infl_pob; shortmetrics=false, prefix="")
    _prefix = prefix == "" ? "" : prefix * "_"
    T = size(tray_infl, 1)
    K = size(tray_infl, 3)
    
    # MSE 
    mse = _mse(tray_infl, tray_infl_pob)
    shortmetrics && return Dict(Symbol(_prefix, "mse") => mse) # only MSE if shortmetrics=true
    
    # Error distributions 
    err_dist = tray_infl .- tray_infl_pob
    # Squared error distribution
    mse_dist = vec(mean(x -> x^2, err_dist, dims=1))
    sq_err_dist = err_dist .^ 2
    # Standard deviation of the MSE distribution for the full period
    std_mse_dist = std(mse_dist, mean=mse) 
    
    # Standard error of simulation of the obtained average value
    mse_std_error = std_mse_dist / sqrt(K)
    # mse_std_error = std(sq_err_dist, mean=mse) / sqrt(T * K)
    
    # Standard deviation of the squared error ~ all periods and realizations
    std_sqerr_dist = std(sq_err_dist, mean=mse)
    
    # RMSE, MAE, ME
    rmse = mean(sqrt, mse_dist)
    mae = mean(abs, err_dist)
    me = mean(err_dist)
    
    # Huber loss ~ combines the properties of MSE and MAE
    huber = mean(huber_loss, err_dist)

    # Correlation 
    corr_dist = first.(cor.(eachslice(tray_infl, dims=3), Ref(tray_infl_pob)))
    corr = mean(corr_dist) 

    ## Additive decomposition of the MSE

    # Bias^2
    me_dist = vec(mean(err_dist, dims=1))
    mse_bias = mean(x -> x^2, me_dist)

    # Variance component
    s_param = std(tray_infl_pob, corrected=false)
    s_tray_infl = vec(std(tray_infl, dims=1, corrected=false))
    mse_var = mean(s -> (s - s_param)^2, s_tray_infl)

    # Correlation component
    mse_cov_dist = @. 2 * (1 - corr_dist) * s_param * s_tray_infl
    mse_cov = mean(mse_cov_dist)

    # Dictionary of metrics to return
    Dict(
        Symbol(_prefix, "mse") => mse, 
        Symbol(_prefix, "mse_std_error") => mse_std_error, 
        Symbol(_prefix, "std_mse_dist") => std_mse_dist, 
        Symbol(_prefix, "std_sqerr_dist") => std_sqerr_dist, 
        Symbol(_prefix, "rmse") => rmse, 
        Symbol(_prefix, "mae") => mae, 
        Symbol(_prefix, "me") => me, 
        Symbol(_prefix, "absme") => abs(me), 
        Symbol(_prefix, "corr") => corr, 
        Symbol(_prefix, "huber") => huber,
        Symbol(_prefix, "mse_bias") => mse_bias, 
        Symbol(_prefix, "mse_var") => mse_var, 
        Symbol(_prefix, "mse_cov") => mse_cov,
        Symbol(_prefix, "T") => T
    )
end


# Huber loss
function huber_loss(x::Real; a=1)
    if abs(x) <= a 
        return x^2 / 2
    else 
        return a*(abs(x) - a/2)
    end
end


# Mean squared error - MSE: average across time and number of
# realizations of the errors between the inflation trajectories tray_infl and
# the parametric inflation trajectory tray_pob
function _mse(tray_infl, tray_infl_pob)
    mean(x -> x^2, tray_infl .- tray_infl_pob)
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
    tray_infl_comb = sum(tray_infl .* w', dims=2)
    eval_metrics(tray_infl_comb, tray_infl_param; kwargs...)
end