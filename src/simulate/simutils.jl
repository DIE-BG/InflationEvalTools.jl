# Simulation functions for SimConfig

# This function can evaluate only one inflation measure
"""
    compute_lowlevel_sim(data::CountryStructure, config::SimConfig;
            rndseed = DEFAULT_SEED,
            shortmetrics = false,
            showprogress = false) -> (Dict, Array{<:AbstractFloat, 3})

Generate the parametric (population) trajectory, simulated inflation
trajectories and evaluation metrics using a [`SimConfig`](@ref).

Returns a tuple `(metrics, traj_infl)` where `metrics` is a dictionary of
evaluation measures and `traj_infl` is a 3‑D array with dimensions
`(T, 1, K)` (T = time steps, K = number of bootstrap realizations).

Arguments
- `data::CountryStructure`: country dataset container used for resampling and
    trend estimation.
- `config::SimConfig`: configuration with `inflfn`, `resamplefn`, `trendfn`,
    `paramfn`, `nsim` and evaluation periods.

Keyword arguments
- `rndseed`: integer seed for the random generator (default = `DEFAULT_SEED`).
- `shortmetrics`: when true compute a reduced set of metrics (faster / lower
    memory footprint). Default `false`.
- `showprogress`: show progress bar during simulation generation (default
    `false`).

See also [`eval_metrics`](@ref) for details on the metric names stored in
`metrics`.
"""
function compute_lowlevel_sim(
    data::CountryStructure, config::SimConfig;
    rndseed=DEFAULT_SEED,
    shortmetrics=false,
    showprogress=false,
)

    # Get data up to the configuration date
    data_eval = data[config.traindate]

    # Get the population trend inflation trajectory
    param = InflationParameter(config.paramfn, config.resamplefn, config.trendfn)
    traj_infl_pob = param(data_eval)

    @info "B-TIMA assessment simulation" measure = measure_name(config.inflfn) resample = method_name(config.resamplefn) trend = method_name(config.trendfn) assessment = measure_name(config.paramfn) simulations = config.nsim traindate = config.traindate periods = config.evalperiods

    # Generate the simulated inflation trajectories
    traj_infl = pargentrajinfl(
        config.inflfn, # inflation function
        config.resamplefn, # resampling function
        config.trendfn, # trend function
        data_eval; # evaluation data
        rndseed,
        numreplications=config.nsim,
        showprogress,
    )

    # Evaluation metrics in each subperiod of config
    metrics = mapreduce(merge, config.evalperiods) do period
        mask = eval_periods(data_eval, period)
        prefix = period_tag(period)
        metrics = @views eval_metrics(traj_infl[mask, :, :], traj_infl_pob[mask]; shortmetrics, prefix)
        metrics
    end
    # Show the main assessment metrics by default, the RMSE
    @info "Assessment metrics:" filter(t -> contains(string(t), "rmse"), metrics)...

    # Return these values
    return metrics, traj_infl
end

# Function to obtain results dictionary and trajectories from a
# `SimConfig` (and to package metadata + metrics for storage)
"""
        compute_assessment_sim(data::CountryStructure, config::SimConfig;
                        rndseed = DEFAULT_SEED,
                        savetrajectories = false,
                        shortmetrics = false,
                        showprogress = false) -> Dict

Run the low‑level simulation (`compute_lowlevel_sim`), collect evaluation
metrics and return a results dictionary that merges the `SimConfig`
parameters with the computed metrics.

Returns a dictionary `results` that always contains the metrics and config
fields. If `savetrajectories=true` the returned dictionary includes the key
`:trajinfl` containing the simulated trajectories array.

Keyword arguments
- `savetrajectories`: when true add trajectories to the results dictionary
    (in the key `:trajinfl`). Default `false`.
- `shortmetrics`: compute a reduced metrics set when true (default `false`).
- `showprogress`: show progress during trajectory generation (default
    `false`).

Example

```julia-repl
julia> results = compute_assessment_sim(gtdata, config)
```
"""
function compute_assessment_sim(
    data::CountryStructure, config::SimConfig;
    rndseed=DEFAULT_SEED,
    savetrajectories=false,
    shortmetrics=false,
    showprogress=false,
)

    # Run the simulation and get the results
    metrics, trajinfl = compute_lowlevel_sim(
        data, config;
        rndseed,
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


# Function to run a batch of simulations
"""
        run_assessment_batch(data::CountryStructure, dict_list_params, savepath::AbstractString;
                rndseed = DEFAULT_SEED,
                savetrajectories = false,
                shortmetrics = false,
                showprogress = true)

Generate a batch of simulation assessments from a list (or iterable) of
parameter dictionaries. Each element in `dict_list_params` is converted to a
`SimConfig` with `dict2config` and evaluated with
`compute_assessment_sim`. Results are saved to `savepath` using
`DrWatson.savename` so they can later be read by `collect_results`.

Arguments
- `dict_list_params`: iterable of parameter dictionaries (usually created by
    expanding vectors of parameter values).
- `savepath`: directory where per‑run result files will be written.

Keyword arguments
- `savetrajectories`: save trajectories within each result file (default
    `false`).
- `shortmetrics`: compute reduced metrics when true (default `false`).
- `showprogress`: show progress indicator during generation (default
    `true`).

Example

```julia-repl
julia> config_dict = Dict(
        :inflfn => InflationPercentileWeighted.(50:80),
        :resamplefn => resamplefn,
        :trendfn => trendfn,
        :paramfn => paramfn,
        :traindate => Date(2019, 12),
        :nsim => 1000) |> dict_list

run_assessment_batch(gtdata_eval, config_dict, savepath)
```

After the batch completes, use `collect_results(savepath)` to assemble a
`DataFrame` with the stored metrics.
"""
function run_assessment_batch(
    data::CountryStructure, dict_list_params, savepath::AbstractString;
    rndseed=DEFAULT_SEED,
    savetrajectories=false,
    shortmetrics=false,
    showprogress=true,
)

    # Run batch of simulations
    for (i, dict_params) in enumerate(dict_list_params)
        @info "Running simulation $i of $(length(dict_list_params))..."
        config = dict2config(dict_params)
        results = compute_assessment_sim(
            data, config;
            rndseed,
            shortmetrics,
            savetrajectories,
            showprogress
        )

        # Save the results
        # Rsults intended for DrWatson.collect_results
        filename = DrWatson.savename(config, "jld2")
        wsave(joinpath(savepath, filename), tostringdict(results))
    end
    return
end
