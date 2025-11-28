# SimDynamicConfig.jl - Definition of SimDynamicConfig for dynamic trend simulations

"""
    SimDynamicConfig{F, R, T}

Configuration type for simulations using the dynamic random walk trend (`TrendDynamicRW`).
This configuration is similar to `SimConfig` but does not include a `trendfn` field,
as it is implicitly designed for the dynamic trend.

It contains:
- `inflfn`: Inflation function
- `resamplefn`: Resampling function
- `trendfns`: Vector of trend functions.
- `paramfn`: Parametric inflation function (for population trend)
- `nsim`: Number of simulations per fold
- `nfolds`: Number of folds (trend instantiations)
- `traindate`: End date of training set
- `evalperiod`: Evaluation period

## Example
```julia
config = SimDynamicConfig(inflfn, resamplefn, trend_functions, paramfn, 1000, 10, Date(2019,12), CompletePeriod())
```
"""
Base.@kwdef struct SimDynamicConfig{F, R, T <: Vector{<:TrendFunction}}
    # Inflation function
    inflfn::F
    # Resampling function
    resamplefn::R
    # Vector of trend functions
    trendfns::T
    # Parametric inflation function
    paramfn::InflationFunction
    # Number of simulations per fold
    nsim::Int
    # Final evaluation date
    traindate::Date
    # Evaluation period
    evalperiod::AbstractEvalPeriod
end

# Method to show configuration information in the REPL
function Base.show(io::IO, config::SimDynamicConfig)
    println(io, typeof(config))
    println(io, "↳ Inflation function          : ", measure_name(config.inflfn))
    println(io, "↳ Resampling function         : ", method_name(config.resamplefn))
    println(io, "↳ Trend function              : Dynamic Random Walk (Implicit)")
    println(io, "↳ Parametric inflation method : ", measure_name(config.paramfn))
    println(io, "↳ Number of simulations       : ", config.nsim)
    println(io, "↳ Number of folds             : ", length(config.trendfns))
    println(io, "↳ End of training set         : ", Dates.format(config.traindate, DEFAULT_DATE_FORMAT))
    return println(io, "↳ Evaluation period           : ", config.evalperiod)
end

# Extension of allowed types for simulation in DrWatson
DrWatson.default_allowed(::SimDynamicConfig) = (String, Symbol, TimeType, Function, Real)

# Extension of savename for SimDynamicConfig
DrWatson.savename(config::SimDynamicConfig, suffix::String = "jld2"; kwargs...) =
    savename(DrWatson.default_prefix(config), config, suffix; kwargs...)

function DrWatson.savename(prefix::String, config::SimDynamicConfig, suffix::String; kwargs...)
    _prefix = prefix == "" ? "" : prefix * "_"
    _suffix = suffix != "" ? "." * suffix : ""
    period_tag_ = config.evalperiod == CompletePeriod() ? "CompletePeriod" : period_tag(config.evalperiod)

    return _prefix * join(
        [
            measure_tag(config.inflfn), # Inflation function
            method_tag(config.resamplefn), # Resampling function
            "DynamicRW", # Implicit Trend function tag
            measure_tag(config.paramfn), # Parametric inflation function for evaluation
            config.nsim >= 1000 ? string(config.nsim ÷ 1000) * "k" : string(config.nsim), # Number of simulations
            string(length(config.trendfns)), # Number of folds
            Dates.format(config.traindate, COMPACT_DATE_FORMAT),
            period_tag_, # Evaluation period
        ], DEFAULT_CONNECTOR
    ) * _suffix
end

"""
    SimDynamicConfig(params::Dict)

Constructor to create a `SimDynamicConfig` from a dictionary of parameters.
"""
function SimDynamicConfig(params::Dict)
    required_keys = (:inflfn, :resamplefn, :trendfns, :paramfn, :nsim, :traindate, :evalperiod)
    # Check that all required keys are present
    if !all(k -> haskey(params, k), required_keys)
        missing_keys = filter(k -> !haskey(params, k), required_keys)
        error("Missing keys in params dictionary for SimDynamicConfig: $(join(missing_keys, ", "))")
    end
    return SimDynamicConfig(params[:inflfn], params[:resamplefn], params[:trendfns], params[:paramfn], params[:nsim], params[:traindate], params[:evalperiod])
end
