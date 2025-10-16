## Definitions to obtain an inflation parameter

"""
Abstract type to represent inflation parameters
"""
abstract type AbstractInflationParameter{F <: InflationFunction, R <: ResampleFunction, T <: TrendFunction} end

"""
Concrete type to represent an inflation parameter computed with the
inflation function `inflfn`, the resampling method `resamplefn`, and trend
function `trendfn`.

See also: [`ParamTotalCPIRebase`](@ref), [`ParamTotalCPI`](@ref), [`ParamWeightedMean`](@ref)
"""
Base.@kwdef struct InflationParameter{F, R, T} <: AbstractInflationParameter{F, R, T}
    inflfn::F = InflationTotalRebaseCPI()
    resamplefn::R = ResampleSBB(36)
    trendfn::T = TrendRandomWalk()
end

# Method to obtain the parametric trajectory from a CountryStructure
function (param::AbstractInflationParameter)(cs::CountryStructure)
    # Obtain the function to get the parametric (average) data from the resampling method
    paramfn = get_param_function(param.resamplefn)
    # Compute a CountryStructure with parametric (average) data
    param_data = paramfn(cs)
    # Apply the trend
    trended_data = param.trendfn(param_data)
    # Apply the inflation function to obtain the parametric trajectory
    traj_infl_param = param.inflfn(trended_data)

    # Return the parametric inflation trajectory
    return traj_infl_param
end

# Redefine a Base.show method for InflationParameter
function Base.show(io::IO, param::AbstractInflationParameter)
    println(io, typeof(param))
    println(io, "↳ InflationFunction : " * measure_name(param.inflfn))
    println(io, "↳ ResampleFunction  : " * method_name(param.resamplefn))
    println(io, "↳ TrendFunction     : " * method_name(param.trendfn))
    return
end

method_tag(param::InflationParameter) = string("InflParam: [", nameof(param.inflfn), ", ", nameof(param.resamplefn), ", ", nameof(param.trendfn), "]")

"""
    DEFAULT_RESAMPLE_FN

Defines the default resampling function to use in the simulation exercise.
"""
const DEFAULT_RESAMPLE_FN = ResampleScrambleVarMonths()


"""
    DEFAULT_TREND_FN

Defines the default trend function to use in the simulation exercise.
"""
const DEFAULT_TREND_FN = TrendRandomWalk()


"""
    ParamTotalCPIRebase()

Helper function to obtain the configuration of the inflation parameter given
by the synthetic base change CPI inflation function, and the default
resampling method and trend function.
"""
ParamTotalCPIRebase() =
    InflationParameter(InflationTotalRebaseCPI(60), DEFAULT_RESAMPLE_FN, DEFAULT_TREND_FN)

# Function to obtain the parameter with another resampling function and another trend function.
ParamTotalCPIRebase(resamplefn::ResampleFunction, trendfn::TrendFunction) =
    InflationParameter(InflationTotalRebaseCPI(60), resamplefn, trendfn)

"""
    ParamTotalCPI()

Helper function to obtain the configuration of the inflation parameter given
by the CPI inflation function, and the default resampling method and trend
function.
"""
ParamTotalCPI() = InflationParameter(InflationTotalCPI(), DEFAULT_RESAMPLE_FN, DEFAULT_TREND_FN)

# Function to obtain the parameter with another resampling function and another trend function.
ParamTotalCPI(resamplefn::ResampleFunction, trendfn::TrendFunction) =
    InflationParameter(InflationTotalCPI(), resamplefn, trendfn)


"""
    ParamTotalCPILegacyRebase()

Helper function to obtain the configuration of the inflation parameter given
by the synthetic base change CPI inflation function, and the default
resampling method and trend function.
"""
ParamTotalCPILegacyRebase() =
    InflationParameter(InflationTotalRebaseCPI(36, 2), ResampleScrambleVarMonths(), DEFAULT_TREND_FN)

# Function to obtain the parameter with another resampling function and another trend function.
ParamTotalCPILegacyRebase(resamplefn::ResampleFunction, trendfn::TrendFunction) =
    InflationParameter(InflationTotalRebaseCPI(36, 2), resamplefn, trendfn)


"""
    ParamWeightedMean()

Helper function to obtain the configuration of the inflation parameter given
by the interannual weighted mean and the default resampling method.
"""
ParamWeightedMean() = InflationParameter(InflationWeightedMean(), DEFAULT_RESAMPLE_FN, DEFAULT_TREND_FN)

# Function to obtain the parameter with another resampling function and another
# trend function.
ParamWeightedMean(resamplefn::ResampleFunction, trendfn::TrendFunction) =
    InflationParameter(InflationWeightedMean(), resamplefn, trendfn)
