##  ----------------------------------------------------------------------------
#   Test script for the 2025 B-TIMA Extension Methodology
#   ----------------------------------------------------------------------------
using InflationEvalTools
using Test
using CPIDataGT
using Distributions: pdf, Normal
using Statistics: mean, std
using StatsBase: sample, sample!
import Random

# Load GT data
CPIDataGT.load_data()
include("BTIMA_extension_helpers.jl")  # Extension helper functions

# Test data
GT24_test = GTDATA24[Date(2025,1), Date(2025,9)].base[1]

allin(v_sample, v_collection) = all(in.(v_sample, Ref(v_collection)))
nonein(v_sample, v_collection) = !any(in.(v_sample, Ref(v_collection)))

##  ----------------------------------------------------------------------------
#   Tests for the CPI variety match object (CPIVarietyMatchDistribution)
#   ----------------------------------------------------------------------------

## Instantiate objects

code_b10 = "_0111101"
code_b23 = "_0111101"
m = 1

v_10, v_23 = get_obs_month(code_b10, code_b23, m)
v_10, v_23, name_10, name_23 = get_names_obs_month(code_b10, code_b23, m)

# Create the variety object

# A variety is the tuple (CPI base, item number, month) and indexes a match
# between two items in two adjacent CPI datasets.

# Codes and names should be optional

# This returns the variety
var1 = CPIVarietyMatchDistribution(v_10, v_23, m)
@test var1 isa CPIVarietyMatchDistribution
@test var1.prior_variety_name === nothing
@test var1.prior_variety_id === nothing
@test var1.actual_variety_name === nothing
@test var1.actual_variety_id === nothing
var1 = CPIVarietyMatchDistribution(v_10, v_23, m, :synthetic)
@test var1 isa CPIVarietyMatchDistribution
var1 = CPIVarietyMatchDistribution(
    v_10, v_23, m, :synthetic,
    code_b10,
    name_10,
    code_b23,
    name_23,
)
@test var1 isa CPIVarietyMatchDistribution
var1 = CPIVarietyMatchDistribution(
    v_10, v_23, m, InflationEvalTools.synthetic_reweighing,
    code_b10,
    name_10,
    code_b23,
    name_23,
)
@test var1 isa CPIVarietyMatchDistribution


## Basic stats

# This gets the mean and the standard deviation
mean(var1)
std(var1)

manual_mean = sum(var1.weights .* var1.vkdistr) / sum(var1.weights)
@test mean(var1) ≈ manual_mean

n = count(!iszero, var1.weights)
manual_var = (n / (n - 1) * sum(var1.weights)) * sum(var1.weights .* (var1.vkdistr .- var1.expected_value) .^ 2)
manual_std = sqrt(manual_var)
@test isapprox(std(var1), manual_std; atol = 1.0e-4)

# When created, I can inspect the resulting weights
var1.weights
@test sum(var1.weights) ≈ 1

# I should be able to resample from it
sample(var1)
sample(var1, 10)

synth_distr = [v_10..., v_23...]
@test sample(var1) in synth_distr
@test sample(var1) in var1.vkdistr
@test allin(sample(var1, 10), synth_distr)

rng = Random.default_rng()
Random.seed!(rng, 3141592)
sample(rng, var1)
sample(rng, var1, 10)

bootstrap_sample = sample(var1, 100)
@test mean(bootstrap_sample) ≈ -0.06905082f0

# In-place methods
x = zeros(eltype(var1), 100)
sample!(var1, x)
@test rand(x) in var1.vkdistr


## Sample only from prior observations

var2 = CPIVarietyMatchDistribution(v_10, v_23, m, :prior)
@test var2 isa CPIVarietyMatchDistribution
var2 = CPIVarietyMatchDistribution(v_10, v_23, m, InflationEvalTools.prior_reweighing)
@test var2 isa CPIVarietyMatchDistribution
var2 = CPIVarietyMatchDistribution(v_10, v_23, m, :prior, code_b10, name_10, code_b23, name_23)
@test var2 isa CPIVarietyMatchDistribution
var2 = CPIVarietyMatchDistribution(v_10, v_23, m, InflationEvalTools.prior_reweighing, code_b10, name_10, code_b23, name_23)
@test var2 isa CPIVarietyMatchDistribution


# To sample only from the actual observations

var3 = CPIVarietyMatchDistribution(v_10, v_23, m, :actual)
@test var3 isa CPIVarietyMatchDistribution
var3 = CPIVarietyMatchDistribution(v_10, v_23, m, InflationEvalTools.actual_reweighing)
@test var3 isa CPIVarietyMatchDistribution
var3 = CPIVarietyMatchDistribution(v_10, v_23, m, :actual, code_b10, name_10, code_b23, name_23)
@test var3 isa CPIVarietyMatchDistribution
var3 = CPIVarietyMatchDistribution(v_10, v_23, m, InflationEvalTools.actual_reweighing, code_b10, name_10, code_b23, name_23)
@test var3 isa CPIVarietyMatchDistribution


##  ----------------------------------------------------------------------------
#   Tests with the resampling function for the CountryStructure and VarCPIBase
#   objects
#   ----------------------------------------------------------------------------

nperiods = periods(GT24_test)
nitems = items(GT24_test)

# First 99 items var1 : synthetic
# Up to the 350 items, var2 : prior, then var3 : actual
matching_array = [j < 100 ? var1 : (j < 350) ? var2 : var3 for m in 1:12, j in 1:nitems]

# Resampling function with the same number of periods
synth_resampler = ResampleSynthetic(GT24_test, matching_array)

# It should not work with a CountryStructure
@test_throws ErrorException synth_resampler(GTDATA24)

# Resample a VarCPIBase object
varbase_sample = synth_resampler(GT24_test)

# Check synthetic sample
@test allin(varbase_sample.v[:, 1], synth_distr)

# Check prior-only sample
@test allin(varbase_sample.v[:, 349], v_10)
@test nonein(varbase_sample.v[:, 349], v_23)

# Check actual-only sample
@test nonein(varbase_sample.v[:, 400], v_10)
@test allin(varbase_sample.v[:, 400], v_23)


## Extend the sampling for 5 years

synth_resampler = ResampleSynthetic(GT24_test, matching_array, 12*5)
@test synth_resampler.periods == 60

# Resample a VarCPIBase object
varbase_sample = synth_resampler(GT24_test)

# Check synthetic sample
@test allin(varbase_sample.v[:, 1], synth_distr)

# Check prior-only sample
@test allin(varbase_sample.v[:, 349], v_10)
@test nonein(varbase_sample.v[:, 349], v_23)

# Check actual-only sample
@test nonein(varbase_sample.v[:, 400], v_10)
@test allin(varbase_sample.v[:, 400], v_23)


## Incorrect sizes the VarCPIBase objects

synth_resampler = ResampleSynthetic(GT24_test, matching_array, 12*5)

# GT23 has 437 items, while GT24 has 436
@test_throws BoundsError synth_resampler(GT23)
# @test_throws BoundsError synth_resampler(GT10)

## Shifted dates in GT24

synth_resampler = ResampleSynthetic(GT24_test, matching_array)

shifted_GT24 = VarCPIBase(GT24_test.v, GT24_test.w, Date(2025,3):Month(1):Date(2025,11), GT24_test.baseindex)

# It allows applying it to shifted data...
synth_resampler(shifted_GT24)

# It is an error to create a sampler for a shifted data: 
# Matching array starts in Jan
# shifted_GT24 goes from Mar
@test_throws ErrorException ResampleSynthetic(shifted_GT24, matching_array)


## Population data 

synth_resampler = ResampleSynthetic(GT24_test, matching_array)
population_data_fn = get_param_function(synth_resampler)
population_varbase = population_data_fn(GT24_test)

# Number of periods
@test periods(population_varbase) == periods(GT24_test)
# Check synthetic sample
@test all(population_varbase.v[:, 1] .== mean(var1))
# Check prior-only sample
@test all(population_varbase.v[:, 349] .== mean(var2))
# Check actual-only sample
@test all(population_varbase.v[:, 400] .== mean(var3))

synth_resampler = ResampleSynthetic(GT24_test, matching_array, 12*5)
population_data_fn = get_param_function(synth_resampler)
population_varbase = population_data_fn(GT24_test)

# Number of periods
@test periods(population_varbase) == 12*5
# Check synthetic sample
@test all(population_varbase.v[:, 1] .== mean(var1))
# Check prior-only sample
@test all(population_varbase.v[:, 349] .== mean(var2))
# Check actual-only sample
@test all(population_varbase.v[:, 400] .== mean(var3))

