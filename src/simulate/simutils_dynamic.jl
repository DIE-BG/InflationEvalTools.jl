# simutils_dynamic.jl - Simulation functions for SimDynamicConfig

"""
    compute_lowlevel_sim(
        data::CountryStructure, config::SimDynamicConfig;
        rndseed = DEFAULT_SEED,
        trend_rng_type = Xoshiro,
        shortmetrics = false,
        showprogress = false,
    ) -> (Dict, Array{<:AbstractFloat, 3})

Generate the parametric (population) trajectory, simulated inflation
trajectories and evaluation metrics using a [`SimDynamicConfig`](@ref).

This function iterates `config.nfolds` times. In each fold, it calls 
`config.trendfn(rng)` to instantiate a new trend function (expected to be `TrendDynamicRW`), 
where `rng` is a random number generator seeded for reproducibility.
It generates `config.nsim` simulations per fold.
The results are merged:
- Metrics are collected into vectors (one value per fold).
- Trajectories are concatenated along the third dimension (total simulations = `nsim * nfolds`).

Arguments
- `data::CountryStructure`: country dataset container.
- `config::SimDynamicConfig`: configuration.

Keyword arguments
- `rndseed`: integer seed for the random generator of the simulation trajectories (default = `DEFAULT_SEED`).
- `trend_rng_type`: type of the random number generator (default = `StableRNG`).
- `shortmetrics`: when true compute a reduced set of metrics.
- `showprogress`: show progress bar.
"""
function compute_lowlevel_sim(
        data::CountryStructure, config::SimDynamicConfig;
        rndseed = DEFAULT_SEED,
        trend_rng_type = Xoshiro,
        shortmetrics = true,
        showprogress = true,
    )
    # Get data up to the configuration date
    data_eval = data[config.traindate]

    # Initialize containers for results
    # We don't know the exact type of metrics dict values, so we use Any or infer later
    metrics_list = Vector{Dict}(undef, config.nfolds)
    traj_list = Vector{Array{Float32, 3}}(undef, config.nfolds)
    traj_pob_list = Vector{Vector{Float32}}(undef, config.nfolds)

    # Generate all the trend objects
    rng_trend = trend_rng_type(rndseed)
    trendfns = [config.trendfn(rng_trend) for _ in 1:config.nfolds]

    # Progress bar
    if showprogress
        p = Progress(
            config.nfolds, 
            dt = 0.5, 
            desc = "Running simulations with different realizations of the trend...",
        )
    end

    for i in 1:config.nfolds
        # Set seed for this fold to ensure reproducibility and independence
        # We use different seeds for trend generation and simulation
        sim_seed = rndseed + i + config.nfolds

        # Get the trend function for this fold
        trendfn = trendfns[i]

        # Get the population trend inflation trajectory for this fold
        param = InflationParameter(config.paramfn, config.resamplefn, trendfn)
        traj_infl_pob = param(data_eval)

        # Generate the simulated inflation trajectories for this fold
        traj_infl = pargentrajinfl(
            config.inflfn,
            config.resamplefn,
            trendfn,
            data_eval;
            rndseed = sim_seed,
            numreplications = config.nsim,
            showprogress = false # Don't show inner progress
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
        traj_pob_list[i] = traj_infl_pob

        # Update progress bar if needed
        if showprogress
            next!(p)
        end
    end

    return @dict(metrics_list, traj_list, traj_pob_list, trendfns)
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
        data::CountryStructure, config::SimDynamicConfig;
        rndseed = DEFAULT_SEED,
        trend_rng_type = Xoshiro,
        savetrajectories = false,
        shortmetrics = false,
        showprogress = false
    ) -> Dict

Run the low‑level simulation (`compute_lowlevel_sim`) for dynamic configuration.
"""
function compute_assessment_sim(
        data::CountryStructure, config::SimDynamicConfig;
        rndseed = DEFAULT_SEED,
        trend_rng_type = Xoshiro,
        savetrajectories = false,
        shortmetrics = true,
        showprogress = true,
        verbose = true,
    )

    # Run the simulation and get the results
    results = compute_lowlevel_sim(
        data, config;
        rndseed,
        trend_rng_type,
        shortmetrics,
        showprogress,
    )

    # Merge results
    metrics = _merge_metrics(results[:metrics_list])
    traj_infl_all = cat(results[:traj_list]..., dims = 3)
    traj_infl_pob_all = cat(results[:traj_pob_list]..., dims=3) # Concatenate population trajectories
    trendfns = results[:trendfns]

    # Show summary of assessment metrics (e.g. mean RMSE)
    if verbose
        @info "Assessment metrics (mean over folds):"
        for metric in ["rmse", "me", "corr"]
            k = Symbol(
                period_tag(config.evalperiod), 
                config.evalperiod === CompletePeriod() ? "" : "_",
                metric,
            )
            val = mean(metrics[k])
            stderror = std(metrics[k]) / sqrt(config.nfolds)
            @info "$k: $val ± $stderror"
        end
    end

    # Add results to dictionary
    results = merge(struct2dict(config), metrics)
    results[:measure] = CPIDataBase.measure_name(config.inflfn)
    results[:params] = CPIDataBase.params(config.inflfn)
    results[:trendfns] = trendfns
    if savetrajectories
        results[:trajinfl] = traj_infl_all
        results[:trajinfl_pob] = traj_infl_pob_all
    end

    # Return dictionary with results
    return results
end
