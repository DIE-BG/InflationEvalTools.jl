# SimDynamicConfig.jl - Definition of SimDynamicConfig for dynamic trend simulations

"""
    SimDynamicConfig{F, R}

Configuration type for simulations using the dynamic random walk trend (`TrendDynamicRW`).
This configuration is similar to `SimConfig` but does not include a `trendfn` field,
as it is implicitly designed for the dynamic trend.

It contains:
- `inflfn`: Inflation function
- `resamplefn`: Resampling function
- `paramfn`: Parametric inflation function (for population trend)
- `nsim`: Number of simulations per fold
- `nfolds`: Number of folds (trend instantiations)
- `traindate`: End date of training set
- `evalperiod`: Evaluation period

## Example
```julia
config = SimDynamicConfig(inflfn, resamplefn, paramfn, 1000, 10, Date(2019,12), CompletePeriod())
```
"""
Base.@kwdef struct SimDynamicConfig{F, R}
    # Inflation function
    inflfn::F
    # Resampling function
    resamplefn::R
    # Parametric inflation function 
    paramfn::InflationFunction
    # Number of simulations per fold
    nsim::Int
    # Number of folds (trend instantiations)
    nfolds::Int
    # Final evaluation date 
    traindate::Date
    # Evaluation period
    evalperiod::AbstractEvalPeriod
end

# Method to show configuration information in the REPL
function Base.show(io::IO, config::SimDynamicConfig)
    println(io, typeof(config))
    println(io, "↳ Inflation function            : ", measure_name(config.inflfn))
    println(io, "↳ Resampling function           : ", method_name(config.resamplefn))
    println(io, "↳ Trend function                : Dynamic Random Walk (Implicit)")
    println(io, "↳ Parametric inflation method   : ", measure_name(config.paramfn))
    println(io, "↳ Number of simulations         : ", config.nsim)
    println(io, "↳ Number of folds               : ", config.nfolds)
    println(io, "↳ End of training set           : ", Dates.format(config.traindate, DEFAULT_DATE_FORMAT))
    println(io, "↳ Evaluation period             : ", config.evalperiod)
end

# Extension of allowed types for simulation in DrWatson
DrWatson.default_allowed(::SimDynamicConfig) = (String, Symbol, TimeType, Function, Real) 

# Extension of savename for SimDynamicConfig
DrWatson.savename(config::SimDynamicConfig, suffix::String = "jld2"; kwargs...) = 
    savename(DrWatson.default_prefix(config), config, suffix; kwargs...)

function DrWatson.savename(prefix::String, config::SimDynamicConfig, suffix::String; kwargs...)
    _prefix = prefix == "" ? "" : prefix * "_"
    _suffix = suffix != "" ? "." * suffix : ""

    _prefix * join([ 
        measure_tag(config.inflfn), # Inflation function 
        method_tag(config.resamplefn), # Resampling function 
        "DynamicRW", # Implicit Trend function tag
        measure_tag(config.paramfn), # Parametric inflation function for evaluation
        config.nsim >= 1000 ? string(config.nsim ÷ 1000) * "k" : string(config.nsim), # Number of simulations
        string(config.nfolds), # Number of folds
        Dates.format(config.traindate, COMPACT_DATE_FORMAT)
    ], DEFAULT_CONNECTOR) * _suffix 
end

"""
    SimDynamicConfig(params::Dict)

Constructor to create a `SimDynamicConfig` from a dictionary of parameters.
"""
function SimDynamicConfig(params::Dict)
    required_keys = (:inflfn, :resamplefn, :paramfn, :nsim, :nfolds, :traindate, :evalperiod)
    # Check that all required keys are present
    if !all(k -> haskey(params, k), required_keys)
        missing_keys = filter(k -> !haskey(params, k), required_keys)
        error("Missing keys in params dictionary for SimDynamicConfig: $(join(missing_keys, ", "))")
    end
    SimDynamicConfig(params[:inflfn], params[:resamplefn], params[:paramfn], params[:nsim], params[:nfolds], params[:traindate], params[:evalperiod])
end
