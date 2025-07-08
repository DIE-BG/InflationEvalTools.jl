# Simulation functions for SimConfig

# This function can evaluate only one inflation measure
"""
    evalsim(data::CountryStructure, config::SimConfig; 
        rndseed = DEFAULT_SEED, 
        short = false) -> (Dict, Array{<:AbstractFloat, 3})

This function generates the parametric trajectory, simulation trajectories,
and evaluation metrics using the [`SimConfig`](@ref) configuration.
Returns `(metrics, tray_infl)`.

The evaluation metrics are returned in the `metrics` dictionary. If
`short=true`, the dictionary contains only the `:mse` key. This short
dictionary is useful for iterative optimization. By default, the complete
metrics dictionary is computed, but this process is more memory intensive.
See also [`eval_metrics`](@ref).

The simulated inflation trajectories are returned in `tray_infl` as a
three-dimensional array of size `(T, 1, K)`, where `T` corresponds to the
computed inflation periods and `K` represents the number of simulation
realizations. The unitary dimension `1` is used to later concatenate the
simulation results, for example, when computing an optimal weighted average
measure.

## Usage

The `evalsim` function receives a `CountryStructure` and an `AbstractConfig`
of type [`SimConfig`](@ref).

### Example

Given a configuration of type `SimConfig` and a dataset
`gtdata_eval`

```
julia> config = SimConfig(
        InflationPercentileEq(69),
        ResampleScrambleVarMonths(),
        TrendRandomWalk(),
        InflationTotalRebaseCPI(36, 2), 10_000, Date(2019,12))
SimConfig{InflationPercentileEq, ResampleScrambleVarMonths, TrendRandomWalk{Float32}}
|─> Inflation function              : Equiponderated percentile 69.0
|─> Resampling function             : Bootstrap IID by month of occurrence
|─> Trend function                  : Random walk trend
|─> Parametric inflation method     : Year-on-year CPI variation with synthetic base changes (36, 2)
|─> Number of simulations           : 10000
|─> End of training set             : Dec-19
|─> Evaluation periods              : Full period, gt_b00:Dec-01-Dec-10, gt_t0010:Jan-11-Nov-11 and gt_b10:Dec-11-Dec-20
```

we can run a simulation with the parameters in `config` with:

```julia-repl
julia> results, tray_infl = evalsim(gtdata, config)
┌ Info: Inflation measure evaluation
│   measure = "Equiponderated percentile 69.0"
│   resampling = "Bootstrap IID by month of occurrence"
│   trend = "Random walk trend"
│   evaluation = "Year-on-year CPI variation with synthetic base changes (36, 2)"
│   simulations = 10000
│   traindate = 2019-12-01
└   periods = (Full period, gt_b00:Dec-01-Dec-10, gt_t0010:Jan-11-Nov-11, gt_b10:Dec-11-Dec-20)
... (progress bar)
┌ Info: Evaluation metrics:
│   mse = ...
└   ... (other metrics)
```
"""
function evalsim(data::CountryStructure, config::SimConfig; 
    rndseed = DEFAULT_SEED, 
    short = false)
  
    # Get data up to the configuration date
    data_eval = data[config.traindate]

    # Get the parametric inflation trajectory
    param = InflationParameter(config.paramfn, config.resamplefn, config.trendfn)
    tray_infl_pob = param(data_eval)

    @info "Inflation measure evaluation" medida=measure_name(config.inflfn) remuestreo=method_name(config.resamplefn) tendencia=method_name(config.trendfn) evaluación=measure_name(config.paramfn) simulaciones=config.nsim traindate=config.traindate periodos=config.evalperiods

    # Generate the simulated inflation trajectories
    tray_infl = pargentrayinfl(config.inflfn, # inflation function
        config.resamplefn, # resampling function
        config.trendfn, # trend function
        data_eval, # evaluation data
        rndseed = rndseed, K=config.nsim)
    println()

    # Evaluation metrics in each subperiod of config
    metrics = mapreduce(merge, config.evalperiods) do period 
        mask = eval_periods(data_eval, period)
        prefix = period_tag(period)
        metrics = @views eval_metrics(tray_infl[mask, :, :], tray_infl_pob[mask]; short, prefix)
        metrics 
    end 
    # Filter metrics that start with gt_. The metrics for CompletePeriod()
    # do not contain a prefix and are shown by default.
    @info "Evaluation metrics:" filter(t -> !contains(string(t), "gt_"), metrics)...

    # Return these values
    metrics, tray_infl
end

# Function to obtain results dictionary and trajectories from an
# AbstractConfig
"""
    makesim(data, config::AbstractConfig; 
        rndseed = DEFAULT_SEED
        short = false) -> (Dict, Array{<:AbstractFloat, 3})

## Usage
This function uses the `evalsim` function to generate a set of simulations
based on a `CountryStructure` and an `AbstractConfig`, and generates a
`results` dictionary with all the evaluation metrics and with the information
of the `AbstractConfig` used to generate them. Additionally, it generates an
object with the inflation trajectories. Returns `(metrics, tray_infl)`.

### Examples
`makesim` receives a `CountryStructure` and an `AbstractConfig`, passes it to
`evalsim` and generates the simulations. It stores the metrics and simulation
parameters in the results dictionary, and also returns the simulation
trajectory.

```julia-repl 
julia> results, tray_infl = makesim(gtdata, config)
┌ Info: Inflation measure evaluation
│   measure = "Equiponderated percentile 69.0"
│   resampling = "Bootstrap IID by month of occurrence"
│   trend = "Random walk trend"
│   evaluation = "Year-on-year CPI variation with synthetic base changes (36, 2)"
│   simulations = 10000
│   traindate = 2019-12-01
└   periods = (Full period, gt_b00:Dec-01-Dec-10, gt_t0010:Jan-11-Nov-11, gt_b10:Dec-11-Dec-20)
... (progress bar)
┌ Info: Evaluation metrics:
│   mse = ...
└   ... (other metrics)
```
"""
function makesim(data::CountryStructure, config::SimConfig; 
    rndseed = DEFAULT_SEED, 
    short = false)
        
     # Run the simulation and get the results
    metrics, tray_infl = evalsim(data, config; rndseed, short)

    # Add results to dictionary
    results = merge(struct2dict(config), metrics)
    results[:measure] = CPIDataBase.measure_name(config.inflfn)
    results[:params] = CPIDataBase.params(config.inflfn)

    return results, tray_infl 
end


# Function to run a batch of simulations
"""
    run_batch(data, dict_list_params, savepath; 
        savetrajectories = true, 
        rndseed = DEFAULT_SEED)

The `run_batch` function generates simulation batches based on the
configuration parameter dictionary.

## Usage
The function receives a `CountryStructure`, a dictionary with vectors that
contain simulation parameters, and a directory to store files with the
metrics of each of the generated evaluations.

### Example
We generate a dictionary with configuration parameters for equiponderated
percentiles, from percentile 60 to percentile 80. This generates a
dictionary with 21 different configurations for evaluation.

```julia-repl 
config_dict = Dict(
    :inflfn => InflationPercentileWeighted.(50:80), 
    :resamplefn => resamplefn, 
    :trendfn => trendfn,
    :paramfn => paramfn, 
    :traindate => Date(2019, 12),
    :nsim => 1000) |> dict_list`
``` 

Once `config_dict` is created, we can generate the simulation batch using
`run_batch`.

```julia-repl 
julia> run_batch(gtdata_eval, config_dict, savepath)
... (evaluation progress)
```

Once all simulations have been generated, we can obtain the data using the
`collect_results` function. This function reads the results from `savepath`
and presents them in a `DataFrame`.

```julia-repl 
julia> df = collect_results(savepath)
[ Info: Scanning folder `<savepath>` for result files.
[ Info: Added 31 entries.
...
```
"""
function run_batch(data, dict_list_params, savepath; 
    savetrajectories = true, 
    rndseed = DEFAULT_SEED)

    # Run batch of simulations
    for (i, dict_params) in enumerate(dict_list_params)
        @info "Running simulation $i of $(length(dict_list_params))..."
        config = dict_config(dict_params)
        results, tray_infl = makesim(data, config;
            rndseed = rndseed)
        print("\n\n\n") 
        
        # Save the results
        filename = savename(config, "jld2")
        
        # Evaluation results for collect_results
        wsave(joinpath(savepath, filename), tostringdict(results))
        
        # Save inflation trajectories, tray_infl directory in the save path
        savetrajectories && wsave(joinpath(savepath, "tray_infl", filename), "tray_infl", tray_infl)
    end

end


