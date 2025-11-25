# simutils_dynamic.jl - Simulation functions for SimDynamicConfig

"""
    compute_lowlevel_sim(
        data::CountryStructure, config::SimDynamicConfig, trend_constructor::Function;
        rndseed = DEFAULT_SEED,
        rng_type = StableRNG,
        shortmetrics = false,
        showprogress = false,
    ) -> (Dict, Array{<:AbstractFloat, 3})

Generate the parametric (population) trajectory, simulated inflation
trajectories and evaluation metrics using a [`SimDynamicConfig`](@ref).

This function iterates `config.nfolds` times. In each fold, it calls 
`trend_constructor(rng)` to instantiate a new trend function (expected to be `TrendDynamicRW`), 
where `rng` is a random number generator seeded for reproducibility.
It generates `config.nsim` simulations per fold.
The results are merged:
- Metrics are collected into vectors (one value per fold).
- Trajectories are concatenated along the third dimension (total simulations = `nsim * nfolds`).

Arguments
- `data::CountryStructure`: country dataset container.
- `config::SimDynamicConfig`: configuration.
- `trend_constructor::Function`: function that receives an `AbstractRNG` and returns a trend function.

Keyword arguments
- `rndseed`: integer seed for the random generator (default = `DEFAULT_SEED`).
- `rng_type`: type of the random number generator (default = `StableRNG`).
- `shortmetrics`: when true compute a reduced set of metrics.
- `showprogress`: show progress bar.
"""
function compute_lowlevel_sim(
    data::CountryStructure, config::SimDynamicConfig, trend_constructor::Function;
    rndseed=DEFAULT_SEED,
    rng_type=StableRNG,
    shortmetrics=false,
    showprogress=false,
)
    # Get data up to the configuration date
    data_eval = data[config.traindate]
    
    # Initialize containers for results
    # We don't know the exact type of metrics dict values, so we use Any or infer later
    metrics_list = Vector{Dict}(undef, config.nfolds)
    traj_list = Vector{Array{Float32, 3}}(undef, config.nfolds)

    # Progress bar
    if showprogress
        p = Progress(config.nfolds, dt=0.5, desc="Running dynamic simulations...")
    end

    for i in 1:config.nfolds
        # Set seed for this fold to ensure reproducibility and independence
        # We use different seeds for trend generation and simulation
        trend_seed = rndseed + i
        sim_seed = rndseed + i + config.nfolds

        # RNG for trend construction
        rng_trend = rng_type(trend_seed)

        # Instantiate dynamic trend
        trendfn = trend_constructor(rng_trend)

        # Get the population trend inflation trajectory for this fold
        param = InflationParameter(config.paramfn, config.resamplefn, trendfn)
        traj_infl_pob = param(data_eval)

        # Generate the simulated inflation trajectories for this fold
        traj_infl = pargentrajinfl(
            config.inflfn,
            config.resamplefn,
            trendfn,
            data_eval;
            rndseed=sim_seed,
            numreplications=config.nsim,
            showprogress=false # Don't show inner progress
        )

        # Compute metrics for the single evaluation period
        mask = eval_periods(data_eval, config.evalperiod)
        prefix = period_tag(config.evalperiod)
        
        # Evaluate metrics
        # Note: eval_metrics returns a Dict of scalars
        m = @views eval_metrics(traj_infl[mask, :, :], traj_infl_pob[mask]; shortmetrics, prefix)
        
        # Save the results for this fold
        metrics_list[i] = m
        traj_list[i] = traj_infl

        # Update progress bar if needed
        if showprogress
            next!(p)
        end
    end

    # Merge results
    metrics = _merge_metrics(metrics_list)
    traj_infl_all = cat(traj_list..., dims=3)

    # Show summary of assessment metrics (e.g. mean RMSE)
    rmse_keys = filter(k -> contains(string(k), "rmse"), keys(metrics))
    if !isempty(rmse_keys)
        @info "Assessment metrics (Mean over folds):" 
        for k in rmse_keys
            val = mean(metrics[k])
            stderror = std(metrics[k]) / sqrt(config.nfolds)
            @info "$k: $val ± $stderror"
        end
    end

    return metrics, traj_infl_all
end

"""
    _merge_metrics(metrics_list::Vector{Dict})

Merge a list of metric dictionaries into a single dictionary where values are vectors.
"""
function _merge_metrics(metrics_list::Vector{Dict})
    isempty(metrics_list) && return Dict()
    
    # Assume all dicts have the same keys
    keys_set = keys(metrics_list[1])
    merged = Dict{Symbol, Any}()
    
    for k in keys_set
        # Collect values for key k from all dicts
        vals = [d[k] for d in metrics_list]
        merged[k] = vals
    end
    
    return merged
end

"""
    compute_assessment_sim(
        data::CountryStructure, config::SimDynamicConfig, trend_constructor::Function;
        rndseed = DEFAULT_SEED,
        rng_type = StableRNG,
        savetrajectories = false,
        shortmetrics = false,
        showprogress = false
    ) -> Dict

Run the low‑level simulation (`compute_lowlevel_sim`) for dynamic configuration.
"""
function compute_assessment_sim(
    data::CountryStructure, config::SimDynamicConfig, trend_constructor::Function;
    rndseed=DEFAULT_SEED,
    rng_type=StableRNG,
    savetrajectories=false,
    shortmetrics=false,
    showprogress=false,
)

    # Run the simulation and get the results
    metrics, trajinfl = compute_lowlevel_sim(
        data, config, trend_constructor;
        rndseed,
        rng_type,
        shortmetrics,
        showprogress,
    )

    # Add results to dictionary
    results = merge(struct2dict(config), metrics)
    results[:measure] = CPIDataBase.measure_name(config.inflfn)
    results[:params] = CPIDataBase.params(config.inflfn)
    savetrajectories && (results[:trajinfl] = trajinfl)

    # Return dictionary with results
    return results
end
