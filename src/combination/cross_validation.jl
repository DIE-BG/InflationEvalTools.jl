## Function to obtain cross-validation error using CrossEvalConfig 

"""
    crossvalidate(
        weightsfunction::Function,
        crossvaldata::Dict{String}, 
        config::CrossEvalConfig = crossvaldata["config"];
        show_status::Bool = true,
        print_weights::Bool = true, 
        return_weights::Bool = false,
        metrics::Vector{Symbol} = [:mse], 
        train_start_date::Date = Date(2000, 12), 
        components_mask = Colon(), 
        add_intercept::Bool = false) -> (cv_results::Matrix [, weights::Vector]) 

Carries out a process of linear combination of inflation measures and
validation evaluation over future subperiods. The inflation measures to be
combined are generated with the `config` configuration of type
[`CrossEvalConfig`](@ref), as well as the simulation parameters and evaluation
periods.

The `crossvaldata` dictionary contains the inflation trajectories, the
parametric trajectory, and the dates of each combination and evaluation period.
The `crossvaldata` dictionary is produced by [`makesim`](@ref) for a
`CrossEvalConfig`. This is done so that the inflation trajectories are
precomputed, as it would be very costly to generate them on the fly.

The `weightsfunction` function receives a tuple `(tray_infl, tray_param)` and
obtains combination weights for the measures in `tray_infl`. For example, you
can directly use the [`combination_weights`](@ref) function, or an anonymous
function built with [`ridge_combination_weights`](@ref) or
[`lasso_combination_weights`](@ref).

Optional parameters:  
- `show_status::Bool = true`: shows information about each weight adjustment
  period (training subperiod) and results of the metrics in the validation
  subperiods.
- `print_weights::Bool = true`: indicates whether to print the weight vectors
  obtained in each training and evaluation iteration.
- `return_weights::Bool = false`: indicates whether to return the weight vector
  of the last period.
- `metrics::Vector{Symbol} = [:mse]`: vector of metrics to report in each
  training and evaluation iteration. The metrics are obtained by
  [`eval_metrics`](@ref).
- `train_start_date::Date = Date(2000, 12)`: start date for the training
  subperiod of the data over which the combination weights are obtained.
- `components_mask = (:)`: mask to apply to the columns of `tray_infl` in the
  combination and evaluation. Used to exclude one or more measures from the
  weight adjustment and out-of-sample evaluation process.
- `add_intercept::Bool = false`: indicates whether to add a column of ones to
  the inflation trajectories to combine. If the `ensemblefn` of `config`
  contains an `InflationConstant` as the first entry, this argument is not
  necessary. Used to obtain an intercept in the linear combination of inflation
  trajectories so that the weights obtained from the combination represent
  variations around this intercept.
"""
function crossvalidate(
    weightsfunction::Function,
    crossvaldata::Dict{String}, 
    config::CrossEvalConfig = crossvaldata["config"]; 
    show_status::Bool = true,
    print_weights::Bool = true, 
    return_weights::Bool = false,
    metrics::Vector{Symbol} = [:mse], 
    train_start_date::Date = Date(2000, 12), 
    components_mask = Colon(), 
    add_intercept::Bool = false) 

    local w
    folds = length(config.evalperiods)
    cv_results = zeros(Float32, folds, length(metrics))

    # Obtener parámetro de inflación 
    for (i, evalperiod) in enumerate(config.evalperiods)
    
        @debug "Ejecutando iteración $i de validación cruzada" evalperiod 

        # Obtener los datos de entrenamiento y validación 
        traindate = evalperiod.startdate - Month(1)
        cvdate = evalperiod.finaldate
        
        train_tray_infl = crossvaldata[_getkey("infl", traindate)]
        train_tray_infl_param = crossvaldata[_getkey("param", traindate)]
        train_dates = crossvaldata[_getkey("dates", traindate)]
        cv_tray_infl = crossvaldata[_getkey("infl", cvdate)]
        cv_tray_infl_param = crossvaldata[_getkey("param", cvdate)]
        cv_dates = crossvaldata[_getkey("dates", cvdate)]

        # Si se agrega intercepto, agregar 1's a las trayectorias. Esto puede
        # alterar el significado de components_mask
        if add_intercept
            train_tray_infl = add_ones(train_tray_infl)
            cv_tray_infl = add_ones(cv_tray_infl)
        end

        # Máscara de períodos para ajustar los ponderadores. Los ponderadores se
        # ajustan a partir de train_start_date
        weights_train_period_mask = train_dates .>= train_start_date

        # Obtener ponderadores de combinación lineal con weightsfunction 
        w = @views weightsfunction(
            train_tray_infl[weights_train_period_mask, components_mask, :], 
            train_tray_infl_param[weights_train_period_mask])

        # Máscara de períodos de evaluación 
        periods_mask = evalperiod.startdate .<= cv_dates .<= evalperiod.finaldate

        # Obtener métrica de evaluación en subperíodo de CV 
        cv_metrics = @views combination_metrics(
            cv_tray_infl[periods_mask, components_mask, :], 
            cv_tray_infl_param[periods_mask], 
            w)
        cv_results[i, :] = get.(Ref(cv_metrics), metrics, 0)

        show_status && @info "Evaluación ($i/$folds):" train_start_date evalperiod traindate cv_results[i]
        print_weights && println(w)
    
    end

    # Retornar ponderaciones si es seleccionado 
    return_weights && return cv_results, w
    # Retornar métricas de validación cruzada
    cv_results
end


function _getkey(prefix, date) 
    fmt = dateformat"yy" 
    prefix * "_" * Dates.format(date, fmt)
end


"""
    add_ones(tray_infl) -> Array{<:AbstractFloat, 3}

Adds an intercept to the trajectory cube in the first column. If the
input dimensions of `tray_infl` are `(T, n, K)`, this function returns an
array with dimensions `(T+1, n, K)`.
"""
function add_ones(tray_infl)
    T, _, K = size(tray_infl)
    hcat(ones(eltype(tray_infl), T, 1, K), tray_infl)
end