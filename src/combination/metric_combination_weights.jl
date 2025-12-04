# Support function to evaluate the metric of the linear combination
function eval_combination(
        tray_infl::AbstractArray{F, 3}, tray_infl_param, w;
        metric::Symbol = :corr,
        sum_abstol::AbstractFloat = 1.0f-2
    ) where {F}

    n = size(tray_infl, 2)
    s = metric == :corr ? -1 : 1 # sign for objective metric

    bp = 2 * one(F) # base penalty for signs
    restp = 5 * one(F) # penalty for sum constraint

    penalty = zero(F)
    for i in 1:n
        if w[i] < 0
            penalty += bp - 2(w[i])
        end
    end
    if !(abs(sum(w) - 1) < sum_abstol)
        penalty += restp + 2 * abs(sum(w) - 1)
    end
    penalty != 0 && return penalty

    # Compute the metric and return its value
    obj = combination_metrics(tray_infl, tray_infl_param, w)[metric]
    return s * obj
end


"""
    metric_combination_weights(tray_infl::AbstractArray{F, 3}, tray_infl_param; 
        metric::Symbol = :corr, 
        w_start = nothing, 
        x_abstol::AbstractFloat = 1f-2, 
        f_abstol::AbstractFloat = 1f-4, 
        max_iterations::Int = 1000) where F

Obtains optimal combination weights for the metric `metric` through
an iterative approximation to the optimization problem of that metric for the
linear combination of inflation estimators in `tray_infl` using the parametric
inflation trajectory `tray_infl_param`.

Optional parameters: 
- `metric::Symbol = :corr`: metric to optimize. If it is linear correlation,
  the metric is maximized. The rest of the metrics are minimized. See also
  [`eval_metrics`](@ref).
- `w_start = nothing`: initial search weights. Typically, a vector of floating
  point values.
- `x_abstol::AbstractFloat = 1f-2`: maximum absolute deviation of the weights. 
- `f_abstol::AbstractFloat = 1f-4`: maximum absolute deviation in the cost
  function.
- `sum_abstol::AbstractFloat = 1f-2`: maximum permissible absolute deviation in
  the sum of weights, with respect to one.
- `max_iterations::Int = 1000`: maximum number of iterations. 

Returns a vector with the weights associated with each estimator in the
columns of `tray_infl`.

See also: [`combination_weights`](@ref), [`ridge_combination_weights`](@ref),
[`share_combination_weights`](@ref), [`elastic_combination_weights`](@ref).
"""
function metric_combination_weights(
        tray_infl::AbstractArray{F, 3}, tray_infl_param;
        metric::Symbol = :corr,
        w_start = nothing,
        x_abstol::AbstractFloat = 1.0f-2,
        f_abstol::AbstractFloat = 1.0f-4,
        sum_abstol::AbstractFloat = 1.0f-4,
        max_iterations::Int = 1000
    ) where {F}

    # Number of weights
    n = size(tray_infl, 2)

    # Initial point
    if isnothing(w_start)
        w0 = ones(F, n) / n
    else
        w0 = w_start
    end

    # Objective function closure
    objectivefn = w -> eval_combination(tray_infl, tray_infl_param, w; metric, sum_abstol)
    # Iterative optimization
    optres = Optim.optimize(
        objectivefn, # Objective function
        zeros(F, n), ones(F, n), # Bounds
        w0, # Initial search point
        Optim.NelderMead(), # Optimization method
        Optim.Options(
            x_abstol = x_abstol, f_abstol = f_abstol,
            show_trace = true, extended_trace = true,
            iterations = max_iterations
        )
    )

    # Get the weights
    wf = Optim.minimizer(optres)
    return wf
end
