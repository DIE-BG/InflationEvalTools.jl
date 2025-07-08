# SimConfig.jl - Definition of container types for simulation parameters

"""
    abstract type AbstractConfig{F <: InflationFunction, R <:ResampleFunction, T <:TrendFunction} end

`AbstractConfig` is an abstract type to represent simulation variants that generally use an
inflation function `InflationFunction`, a resampling function `ResampleFunction`, and a
trend function `TrendFunction`. It contains the general scheme of the simulation.
"""
abstract type AbstractConfig{F <: InflationFunction, R <:ResampleFunction, T <:TrendFunction} end

"""
    SimConfig{F, R, T} <:AbstractConfig{F, R, T}

Concrete type that contains a base configuration to generate simulations
using all data as the training set. Receives an inflation function
`InflationFunction`, a resampling function `ResampleFunction`, a trend
function `TrendFunction`, an evaluation inflation function [`paramfn`], and
the desired number of simulations [`nsim`].

## Example
Considering the following instances of inflation, resampling, trend, and
evaluation inflation functions:

```
julia> percEq = InflationPercentileEq(80);

julia> resamplefn = ResampleSBB(36);

julia> trendfn = TrendRandomWalk();

julia> paramfn = InflationWeightedMean();
```

We generate a `SimConfig` configuration with 1000 simulations, with default evaluation periods:
- `CompletePeriod()`, 
- `GT_EVAL_B00`, 
- `GT_EVAL_T0010` and  
- `GT_EVAL_B10`

```
julia> config = SimConfig(percEq, resamplefn, trendfn, paramfn, 1000, Date(2019,12))
SimConfig{InflationPercentileEq, ResampleSBB, TrendRandomWalk{Float32}}
|─> Inflation function            : Percentil equiponderado 80.0
|─> Resampling function           : Block bootstrap estacionario con bloque esperado 36
|─> Trend function                : Tendencia de caminata aleatoria
|─> Parametric inflation method   : Media ponderada interanual
|─> Number of simulations         : 1000
|─> End of training set           : Dec-19
|─> Evaluation periods            : Complete period, gt_b00:Dec-01-Dec-10, gt_t0010:Jan-11-Nov-11 y gt_b10:Dec-11-Dec-20
```

To generate a configuration with specific periods, you can provide the collection of periods of interest:

```
julia> config2 = SimConfig(percEq, resamplefn, trendfn, paramfn, 1000, Date(2019,12),
       (CompletePeriod(), EvalPeriod(Date(2008,1), Date(2009,12), "fincrisis")))
SimConfig{InflationPercentileEq, ResampleSBB, TrendRandomWalk{Float32}}
|─> Inflation function            : Percentil equiponderado 80.0
|─> Resampling function           : Block bootstrap estacionario con bloque esperado 36
|─> Trend function                : Tendencia de caminata aleatoria
|─> Parametric inflation method   : Media ponderada interanual
|─> Number of simulations         : 1000
|─> End of training set           : Dec-19
|─> Evaluation periods            : Complete period y fincrisis:Jan-08-Dec-09
```
"""
Base.@kwdef struct SimConfig{F, R, T} <:AbstractConfig{F, R, T}
    # Inflation function
    inflfn::F
    # Resampling function
    resamplefn::R
    # Trend function
    trendfn::T
    # Parametric inflation function 
    paramfn::InflationFunction
    # Number of simulations
    nsim::Int  
    # Final evaluation date 
    traindate::Date
    # Collection of evaluation period(s), by default the complete period 
    evalperiods = (CompletePeriod(),)
end

# Constructor with default evaluation periods for Guatemala
SimConfig(inflfn, resamplefn, trendfn, paramfn, nsim, traindate) = 
    SimConfig(inflfn, resamplefn, trendfn, paramfn, nsim, traindate, 
    # Default period configuration
    (CompletePeriod(), GT_EVAL_B00, GT_EVAL_T0010, GT_EVAL_B10))
    



"""
    CrossEvalConfig{F, R, T} <:AbstractConfig{F, R, T}
    CrossEvalConfig(ensemblefn, resamplefn, trendfn, paramfn, nsim, evalperiods)

`CrossEvalConfig` is a concrete type that contains the base configuration to
generate simulations using a set of inflation functions to be combined.

It receives a
- ensemble inflation function `EnsembleFunction`, 
- a resampling function `ResampleFunction`, 
- a trend function `TrendFunction`, 
- the number of simulations to perform `nsim`, 
- a period (or set of periods) of evaluation [`EvalPeriod`](@ref) in
  which evaluation metrics for cross-validation will be obtained. The
  training period is considered from the start of the sample up to the
  period prior to each given evaluation period.

## Example

Considering a set of inflation, resampling, trend, and parametric inflation functions: 

```
julia> ensemblefn = EnsembleFunction(InflationPercentileEq(72), InflationPercentileWeighted(68));

julia> resamplefn = ResampleSBB(36); 

julia> trendfn = TrendRandomWalk(); 

julia> paramfn = InflationTotalRebaseCPI(60); 
```

We generate a `CrossEvalConfig` configuration with 10000 simulations,
configuring two evaluation periods for the cross-validation methods.

```
julia> config = CrossEvalConfig(ensemblefn, resamplefn, trendfn, paramfn, 10000, 
       (EvalPeriod(Date(2016, 1), Date(2017, 12), "cv1617"), 
       EvalPeriod(Date(2017, 1), Date(2018, 12), "cv1718")))
CrossEvalConfig{InflationTotalRebaseCPI, ResampleSBB, TrendRandomWalk{Float32}}
|─> Inflation function            : ["Percentil equiponderado 72.0", "Percentil ponderado 68.0"]
|─> Resampling function           : Block bootstrap estacionario con bloque esperado 36
|─> Trend function                : Tendencia de caminata aleatoria
|─> Parametric inflation method   : Variación interanual IPC con cambios de base sintéticos (60, 0)
|─> Number of simulations         : 10000
|─> Evaluation periods            : cv1617:Jan-16-Dec-17 y cv1718:Jan-17-Dec-18
```
"""
Base.@kwdef struct CrossEvalConfig{F, R, T} <:AbstractConfig{F, R, T}
    # Set of inflation functions to obtain trajectories to combine 
    inflfn::EnsembleFunction
    # Resampling function
    resamplefn::R
    # Trend function
    trendfn::T
    # Parametric inflation function 
    paramfn::F
    # Number of simulations to perform 
    nsim::Int
    # Collection of evaluation period(s), by default the complete period 
    evalperiods::Union{EvalPeriod, Vector{EvalPeriod}, NTuple{N, EvalPeriod} where N}
end

# Configurations to show function names in savename
Base.string(inflfn::InflationFunction) = measure_tag(inflfn)
Base.string(resamplefn::ResampleFunction) = method_tag(resamplefn)
Base.string(trendfn::TrendFunction) = method_tag(trendfn)

# Method to show configuration information in the REPL
function Base.show(io::IO, config::AbstractConfig)
    println(io, typeof(config))
    println(io, "|─> Inflation function            : ", measure_name(config.inflfn))
    println(io, "|─> Resampling function           : ", method_name(config.resamplefn))
    println(io, "|─> Trend function                : ", method_name(config.trendfn))
    println(io, "|─> Parametric inflation method   : ", measure_name(config.paramfn))
    println(io, "|─> Number of simulations         : ", config.nsim)
    if hasproperty(config, :traindate)
        println(io, "|─> End of training set           : ", Dates.format(config.traindate, DEFAULT_DATE_FORMAT))
    end
    println(io, "|─> Evaluation periods            : ", join(config.evalperiods, ", ", " y "))
end


# Extension of allowed types for simulation in DrWatson
DrWatson.default_allowed(::AbstractConfig) = (String, Symbol, TimeType, Function, Real) 

# Definition of format for saving files related to the configuration
DEFAULT_CONNECTOR = ", "
DEFAULT_EQUALS = "="
DEFAULT_DATE_FORMAT = DateFormat("u-yy")
COMPACT_DATE_FORMAT = DateFormat("uyy")

# Extension of savename for SimConfig
DrWatson.savename(config::SimConfig, suffix::String = "jld2"; kwargs...) = 
    savename(DrWatson.default_prefix(config), config, suffix; kwargs...)

function DrWatson.savename(prefix::String, config::SimConfig, suffix::String; kwargs...)
    _prefix = prefix == "" ? "" : prefix * "_"
    _suffix = suffix != "" ? "." * suffix : ""

    _prefix * join([ 
        measure_tag(config.inflfn), # Inflation function 
        method_tag(config.resamplefn), # Resampling function 
        method_tag(config.trendfn), # Trend function
        measure_tag(config.paramfn), # Parametric inflation function for evaluation
        config.nsim >= 1000 ? string(config.nsim ÷ 1000) * "k" : string(config.nsim), # Number of simulations
        Dates.format(config.traindate, COMPACT_DATE_FORMAT)
    ], DEFAULT_CONNECTOR) * _suffix 
end

# Extension of savename for CrossEvalConfig
DrWatson.savename(config::CrossEvalConfig, suffix::String = "jld2"; kwargs...) = 
    savename(DrWatson.default_prefix(config), config, suffix; kwargs...)

function DrWatson.savename(prefix::String, config::CrossEvalConfig, suffix::String = "jld2"; kwargs...)
    _prefix = prefix == "" ? "" : prefix * "_"
    _suffix = suffix != "" ? "." * suffix : ""
    num_infl_functions = length(config.inflfn.functions)
    num_eval_periods = length(config.evalperiods)
    startdate = minimum(map(p -> p.startdate, config.evalperiods))
    finaldate = maximum(map(p -> p.finaldate, config.evalperiods))

    _prefix * join([
        # Ensemble inflation function denoted by CrossEvalConfig
        "CrossEvalConfig($num_infl_functions, $num_eval_periods)", 
        method_tag(config.resamplefn), # Resampling function 
        method_tag(config.trendfn), # Trend function
        measure_tag(config.paramfn), # Parametric inflation function for evaluation
        config.nsim >= 1000 ? string(config.nsim ÷ 1000) * "k" : string(config.nsim), # Number of simulations
        Dates.format(startdate, COMPACT_DATE_FORMAT) * "-" * Dates.format(finaldate, COMPACT_DATE_FORMAT)
    ], DEFAULT_CONNECTOR) * _suffix
end


# Helper functions


## Method to convert from AbstractConfig to Dictionary
# This is done by the struct2dict() function from DrWatson

"""
    dict_config(params::Dict)

Function to convert a parameter dictionary to `SimConfig` or `CrossEvalConfig`.
"""
function dict_config(params::Dict)
    # CrossEvalConfig contains the field of evaluation periods 
    if (:traindate in keys(params))
        if (:evalperiods in keys(params))
            config = SimConfig(params[:inflfn], params[:resamplefn], params[:trendfn], params[:paramfn], params[:nsim], params[:traindate], params[:evalperiods])
        else 
            config = SimConfig(params[:inflfn], params[:resamplefn], params[:trendfn], params[:paramfn], params[:nsim], params[:traindate])
        end
    else
        config = CrossEvalConfig(params[:inflfn], params[:resamplefn], params[:trendfn], params[:paramfn], params[:nsim], params[:evalperiods])
    end
    config 
end

# Optional method for list of configurations
dict_config(params::AbstractVector) = dict_config.(params)

