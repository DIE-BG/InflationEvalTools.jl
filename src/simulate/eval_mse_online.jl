# eval_mse_online.jl - Generation of online optimization MSE  

"""
    eval_mse_online(config::SimConfig, csdata::CountryStructure; 
        K = 1000, 
        rndseed = DEFAULT_SEED) -> mse

Function to obtain mean squared error evaluation using
[`SimConfig`](@ref) evaluation configuration. The evaluation data must be
provided in `csdata`, with which the comparison parametric trajectory is to be
computed. Returns the MSE as a scalar.

This function can be used to optimize the parameters of different
inflation measures and is more memory efficient than [`pargentrajinfl`](@ref).
"""
function eval_mse_online(config::SimConfig, csdata::CountryStructure; 
    K = 1000, 
    rndseed = DEFAULT_SEED)

    # Create the parameter and obtain the parametric trajectory
    param = InflationParameter(config.paramfn, config.resamplefn, config.trendfn)
    tray_infl_param = param(csdata)

    # Unpack the configuration
    eval_mse_online(config.inflfn, config.resamplefn, config.trendfn, csdata, tray_infl_param; K, rndseed)
end


"""
    eval_mse_online(
        inflfn::InflationFunction,
        resamplefn::ResampleFunction, 
        trendfn::TrendFunction,
        csdata::CountryStructure, 
        tray_infl_param::Vector{<:AbstractFloat}; 
        K = 1000, rndseed = DEFAULT_SEED) -> mse

Function to obtain mean squared error (MSE) evaluation using the
specified configuration. The parametric trajectory
`tray_infl_param` is required to avoid repeatedly computing it in this function. Returns
the MSE as a scalar.
"""
function eval_mse_online(
    inflfn::InflationFunction,
    resamplefn::ResampleFunction, 
    trendfn::TrendFunction,
    csdata::CountryStructure, 
    tray_infl_param::Vector{<:AbstractFloat}; 
    K = 1000, rndseed = DEFAULT_SEED)

    # Trajectory computation task
    mse = @showprogress @distributed (merge) for k in 1:K 
        # Set the seed in the process
        Random.seed!(LOCAL_RNG, rndseed + k)

        # Bootstrap sample of the data
        bootsample = resamplefn(csdata, LOCAL_RNG)
        # Application of the trend function
        trended_sample = trendfn(bootsample)

        # Compute the inflation measure and the MSE
        tray_infl = inflfn(trended_sample)
        sq_err = (tray_infl - tray_infl_param) .^ 2
        o = OnlineStats.Mean(eltype(csdata))
        OnlineStats.fit!(o, sq_err)
    end 

    OnlineStats.value(mse)::eltype(csdata)
end
