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

# Helper functions

## Method to convert from AbstractConfig to Dictionary
# This is done by the struct2dict() function from DrWatson

"""
    dict2config(params::Dict)

Function to convert a parameter dictionary to `SimConfig` or `SimDynamicConfig`.
"""
function dict2config(params::Dict)
    # Check if it is a dynamic configuration
    if haskey(params, :trendfns)
        return SimDynamicConfig(params)
    end

    if (:evalperiods in keys(params))
        config = SimConfig(params[:inflfn], params[:resamplefn], params[:trendfn], params[:paramfn], params[:nsim], params[:traindate], params[:evalperiods])
    else 
        config = SimConfig(params[:inflfn], params[:resamplefn], params[:trendfn], params[:paramfn], params[:nsim], params[:traindate])
    end
    config 
end

# Optional method for list of configurations
dict2config(params::AbstractVector) = dict2config.(params)

