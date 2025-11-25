"""
    InflationEvalTools

Types, functions, and other simulation utilities for the evaluation of inflation measures.
"""
module InflationEvalTools

using CPIDataBase: CPIDataBase, CountryStructure, InflationFunction,
    InflationTotalCPI, VarCPIBase, getunionalltype, infl_dates,
    infl_periods, measure_name, measure_tag, num_measures,
    periods
using Chain: Chain, @chain
using Dates: Dates, Date, DateFormat, Month, TimeType, month
using Distributed: Distributed, @distributed, RemoteChannel
using Distributions: Distributions, Normal, cor, mean, pdf, std
using DrWatson: DrWatson, savename, struct2dict, tostringdict, wsave
using InflationFunctions: InflationFunctions, InflationTotalRebaseCPI,
    InflationWeightedMean
using Ipopt: Ipopt
using JLD2: JLD2, load
using JuMP: JuMP, @constraint, @constraints, @objective, @variable, Model,
    optimize!, set_silent
using ProgressMeter: ProgressMeter, @showprogress, Progress, next!
using Random: AbstractRNG
using Reexport: Reexport
using SharedArrays: SharedArrays, SharedArray, sdata
using StableRNGs: StableRNGs, StableRNG
using LinearAlgebra: LinearAlgebra, mul!
using OnlineStats: OnlineStats


## Default configuration of the seed for the simulation process
"""
    const DEFAULT_SEED

Default seed used for the simulation process and the
reproducibility of the results.
"""
const DEFAULT_SEED = 314159

## Resampling functions for CPI bases

# General methods for resampling functions
export get_param_function, method_name, method_tag
include("resample/ResampleFunction.jl")

# Resampling method using selection of the same months of
# occurrence
export ResampleScrambleVarMonths, ResampleExtendedSVM
export param_scramblevar_fn
include("resample/ResampleScrambleVarMonths.jl")
include("resample/ResampleExtendedSVM.jl")
# Resampling method using selection of the same months of
# occurrence with weighted distributions to maintain correlation in
# the resampling
export ResampleScrambleTrended
include("resample/ResampleScrambleTrended.jl")
# Similar to the previous one, but with individual parameters per base
export ResampleTrended
include("resample/ResampleTrended.jl")
# B-TIMA's Extension methodology: Synthetic resampling using prior information
export CPIVarietyMatchDistribution
export ResampleSynthetic
include("resample/ResampleSynthetic.jl")
# Identity resampler
export ResampleIdentity
include("resample/ResampleIdentity.jl")
# Mixture resampler 
export ResampleMixture
include("resample/ResampleMixture.jl")

## Functions for trend application
export RWTREND
include("trend/RWTREND.jl")

export TrendRandomWalk, TrendAnalytical, TrendExponential, TrendIdentity
include("trend/TrendFunction.jl")

export InflationParameter, ParamTotalCPIRebase, ParamTotalCPI, ParamWeightedMean
export ParamTotalCPILegacyRebase # evaluation parameter 2019
include("param/InflationParameter.jl")

# Types for simulation configuration
export AbstractConfig, SimConfig
export SimDynamicConfig
export CompletePeriod, EvalPeriod, PeriodVector, eval_periods, period_tag
export GT_EVAL_B00, GT_EVAL_B10, GT_EVAL_T0010
include("config/EvalPeriod.jl")
include("config/SimConfig.jl")
include("config/SimDynamicConfig.jl")

## Functions for trajectory generation
export gentrajinfl, pargentrajinfl
include("simulate/gentrajinfl.jl")
include("simulate/pargentrajinfl.jl")

## Functions for evaluation and metrics
export dict2config
export compute_lowlevel_sim, compute_assessment_sim, run_assessment_batch
include("simulate/simutils.jl")
include("simulate/simutils_dynamic.jl")
export eval_metrics, combination_metrics
include("simulate/metrics.jl")
export eval_mse_online # Online MSE evaluation function
export eval_absme_online # Online ABSME evaluation function
export eval_corr_online # Online CORR evaluation function
include("simulate/eval_mse_online.jl")
include("simulate/eval_absme_online.jl")
include("simulate/eval_corr_online.jl")

## Optimal MSE combination of estimators
export combination_weights, average_mats
export ridge_combination_weights, lasso_combination_weights
export share_combination_weights
export elastic_combination_weights
export metric_combination_weights
export absme_combination_weights
include("combination/combination_weights.jl")
include("combination/metric_combination_weights.jl")
include("combination/absme_combination_weights.jl")

## Functions in development

end
