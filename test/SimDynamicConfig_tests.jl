using InflationEvalTools
using InflationFunctions
using CPIDataBase
using CPIDataBase.TestHelpers
using Test
using Dates
using Random
using DrWatson


# Setup test data
cst = getzerocountryst()

# Define components for SimDynamicConfig
inflfn = InflationTotalCPI()
resamplefn = ResampleIdentity()
paramfn = InflationTotalCPI()

# Create a vector of trends
nfolds = 2
trendfns = Vector{TrendDynamicRW}(undef, nfolds)
for i in 1:nfolds
    rng = Xoshiro(123 + i)
    trendfns[i] = TrendDynamicRW(periods(cst), 0.5, 1.0, x -> true; rng = rng)
end

traindate = Date(2020, 12)
evalperiod = CompletePeriod()
nsim = 10

# Test 1: Constructor
config = SimDynamicConfig(
    inflfn,
    resamplefn,
    trendfns,
    paramfn,
    nsim,
    traindate,
    evalperiod,
)

@test config isa SimDynamicConfig
@test config.nsim == nsim
@test length(config.trendfns) == nfolds
@test config.trendfns == trendfns

# Test 2: Constructor from Dict
params_dict = Dict(
    :inflfn => inflfn,
    :resamplefn => resamplefn,
    :trendfns => trendfns,
    :paramfn => paramfn,
    :nsim => nsim,
    :traindate => traindate,
    :evalperiod => evalperiod
)
config_from_dict = SimDynamicConfig(params_dict)
@test config_from_dict isa SimDynamicConfig
@test config_from_dict.nsim == nsim
@test length(config_from_dict.trendfns) == nfolds

# Config from Dict using dict2config
config_from_dict = dict2config(params_dict)
@test config_from_dict isa SimDynamicConfig
@test config_from_dict.nsim == nsim
@test length(config_from_dict.trendfns) == nfolds

# Test 3: savename
# Check that savename generates a string with expected components
sname = DrWatson.savename(config)
@test occursin(measure_tag(inflfn), sname)
@test occursin(method_tag(resamplefn), sname)
@test occursin("DynamicRW", sname)
@test occursin(string(nsim), sname)
@test occursin(string(nfolds), sname)

# Test 4: compute_lowlevel_sim
# We need a CountryStructure with some data. getzerocountryst gives zeros.
# Let's use it, results might be trivial but it checks the pipeline.

results = compute_lowlevel_sim(
    cst, config;
    rndseed = 1234,
    shortmetrics = true,
    showprogress = false
)

@test haskey(results, :metrics_list)
@test haskey(results, :traj_list)
@test haskey(results, :traj_pob_list)
@test haskey(results, :trendfns)

@test length(results[:metrics_list]) == nfolds
@test length(results[:traj_list]) == nfolds
@test length(results[:trendfns]) == nfolds

# Check dimensions of trajectories
# T x 1 x nsim
T = infl_periods(cst)
@test size(results[:traj_list][1]) == (T, 1, nsim)

# Test 5: compute_assessment_sim
assessment_res = compute_assessment_sim(
    cst, config;
    rndseed = 1234,
    shortmetrics = true,
    showprogress = false,
    verbose = false
)

@test assessment_res isa Dict
@test haskey(assessment_res, :measure)
@test haskey(assessment_res, :params)
# Check if metrics are merged (e.g. mse should be present if shortmetrics=true)
# Note: The keys depend on the eval_metrics implementation and period tag.
# CompletePeriod tag is usually "Full".
# Let's check for a key ending in "mse"
keys_str = string.(keys(assessment_res))
@test any(occursin("rmse", k) for k in keys_str)

# Test 6: run_assessment_batch
# Create a temporary directory for results
mktempdir() do savepath
    dict_list = [params_dict]

    run_assessment_batch(
        cst, dict_list, savepath;
        rndseed = 1234,
        shortmetrics = true,
        showprogress = false
    )

    # Check if file was created
    filename = DrWatson.savename(config, "jld2")
    @test isfile(joinpath(savepath, filename))

    # Load and verify
    loaded_res = DrWatson.load(joinpath(savepath, filename))
    @test loaded_res["measure"] == measure_name(inflfn)
end
