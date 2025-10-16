"""
    InflationEvalTools

Types, functions, and other simulation utilities for the evaluation of inflation measures.
"""
module InflationEvalTools

    using DrWatson 
    using Dates
    using CPIDataBase
    using InflationFunctions
    import Random
    using Distributions
    using ProgressMeter
    using Distributed
    using SharedArrays
    using Reexport
    using StableRNGs
    import OnlineStats
    import StatsBase
    using LinearAlgebra: I, det, mul!, dot
    using JuMP, Ipopt
    using Chain
    using JLD2
    import Optim 

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
    export ResampleScrambleVarMonths
    export param_scramblevar_fn
    include("resample/ResampleScrambleVarMonths.jl")
    # Resampling method using selection of the same months of
    # occurrence with weighted distributions to maintain correlation in
    # the resampling
    export ResampleScrambleTrended
    include("resample/ResampleScrambleTrended.jl")
    # Similar to the previous one, but with individual parameters per base 
    export ResampleTrended
    include("resample/ResampleTrended.jl")
    ## Methods to obtain the population datasets of monthly price changes month-to-month variations
    
    ## Functions for trend application
    export RWTREND
    include("trend/RWTREND.jl") 
    
    export TrendRandomWalk, TrendAnalytical, TrendExponential, TrendIdentity
    include("trend/TrendFunction.jl")

    export InflationParameter, ParamTotalCPIRebase, ParamTotalCPI, ParamWeightedMean
    export ParamTotalCPILegacyRebase # evaluation parameter 2019
    include("param/InflationParameter.jl")

    # Types for simulation configuration
    export AbstractConfig, SimConfig, CrossEvalConfig
    export CompletePeriod, EvalPeriod, PeriodVector, eval_periods, period_tag
    export GT_EVAL_B00, GT_EVAL_B10, GT_EVAL_T0010
    include("config/EvalPeriod.jl")
    include("config/SimConfig.jl")
    
    ## Functions for trajectory generation
    export gentrajinfl, pargentrajinfl
    include("simulate/gentrajinfl.jl")
    include("simulate/pargentrajinfl.jl") 
    
    ## Functions for evaluation and metrics   
    export dict2config
    export compute_lowlevel_sim, compute_assessment_sim, run_assessment_batch
    include("simulate/simutils.jl")
    export eval_metrics, combination_metrics
    include("simulate/metrics.jl")
    export eval_mse_online # Online MSE evaluation function
    export eval_absme_online # Online ABSME evaluation function
    export eval_corr_online # Online CORR evaluation function
    include("simulate/eval_mse_online.jl")
    include("simulate/eval_absme_online.jl")
    include("simulate/eval_corr_online.jl")
    include("simulate/cvsimutils.jl") # functions for cross-validation methodology
    

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

    ## Functions for cross-validation
    export add_ones
    export crossvalidate
    include("combination/cross_validation.jl")

    ## Functions in development 

end
