# eval_absme_online.jl - Generation of online optimization ABSME  

"""
    eval_absme_online(config::SimConfig, csdata::CountryStructure; 
        K = 1000, 
        rndseed = DEFAULT_SEED) -> absme

Function to obtain evaluation of the mean absolute error value using
[`SimConfig`](@ref) evaluation configuration. The evaluation data must be
provided in `csdata`, with which the comparison parametric trajectory is to be
computed. Returns the absolute value metric as a scalar.

This function can be used to optimize the parameters of different
inflation measures and is more memory efficient than [`pargentrayinfl`](@ref).
"""
function eval_absme_online(config::SimConfig, csdata::CountryStructure; 
    K = 1000, 
    rndseed = DEFAULT_SEED)

    # Create the parameter and obtain the parametric trajectory
    param = InflationParameter(config.paramfn, config.resamplefn, config.trendfn)
    tray_infl_param = param(csdata)

    # Unpack the configuration
    eval_absme_online(config.inflfn, config.resamplefn, config.trendfn, csdata, tray_infl_param; K, rndseed)
end


"""
    eval_absme_online(
        inflfn::InflationFunction,
        resamplefn::ResampleFunction, 
        trendfn::TrendFunction,
        csdata::CountryStructure, 
        tray_infl_param::Vector{<:AbstractFloat}; 
        K = 1000, rndseed = DEFAULT_SEED) -> absme

Function to obtain evaluation of the mean absolute error value (ABSME)
using the specified configuration. The parametric trajectory
`tray_infl_param` is required to avoid repeatedly computing it in this function. Returns
the ABSME as a scalar.
"""
function eval_absme_online(
    inflfn::InflationFunction,
    resamplefn::ResampleFunction, 
    trendfn::TrendFunction,
    csdata::CountryStructure, 
    tray_infl_param::Vector{<:AbstractFloat}; 
    K = 1000, rndseed = DEFAULT_SEED)

    # Trajectory computation task
    me = @showprogress @distributed (Base.merge) for k in 1:K 
        # Set the seed in the process
        Random.seed!(LOCAL_RNG, rndseed + k)

        # Bootstrap sample of the data
        bootsample = resamplefn(csdata, LOCAL_RNG)
        # Application of the trend function
        trended_sample = trendfn(bootsample)

        # Compute the inflation measure and the ABSME
        tray_infl = inflfn(trended_sample)
        err = (tray_infl - tray_infl_param) 
        o = OnlineStats.Mean(eltype(csdata))
        OnlineStats.fit!(o, err)
    end 

    abs(OnlineStats.value(me))::eltype(csdata)
end
