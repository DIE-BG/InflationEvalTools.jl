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

CPIDataGT.load_data()
include("BTIMA_extension_helpers.jl")  # Extension helper functions

## Instantiate objects

code_b10 = "_0111101"
code_b23 = "_0111101"
m = 1

v_10, v_23 = get_obs_month(code_b10, code_b23, m)
v_10, v_23, name_10, name_23 = get_names_obs_month(code_b10, code_b23, m)

# Normal likelihood (should be an attribute of the variety object)
f(vdistr; a = 0.35, eps = 0.0001) = x -> pdf(Normal(0, a * std(vdistr) + eps), x)

# Create the variety object

# A variety is the tuple (CPI base, item number, month) and indexes a match
# between two items in two adjacent CPI datasets.

# Codes and names should be optional

# This returns the variety
var1 = CPIVarietyMatchDistribution(v_10, v_23, m)
@test var1 isa CPIVarietyMatchDistribution
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
@test all(in.(sample(var1,10), Ref(synth_distr)))

rng = Random.Xoshiro(0)
sample(rng, var1)
sample(rng, var1, 10)

bootstrap_sample = sample(var1, 10000)
@test isapprox(mean(bootstrap_sample), mean(var1); atol = 1.0e-2)

# In-place methods
x = zeros(eltype(var1), 100)
sample!(var1, x)


## Sample only from prior observations

var2 = CPIVarietyMatchDistribution(v_10, v_23, m, :prior)
var2 = CPIVarietyMatchDistribution(v_10, v_23, m, InflationEvalTools.prior_reweighing)
var2 = CPIVarietyMatchDistribution(v_10, v_23, m, :prior, code_b10, name_10, code_b23, name_23)
var2 = CPIVarietyMatchDistribution(v_10, v_23, m, InflationEvalTools.prior_reweighing, code_b10, name_10, code_b23, name_23)
sample(var2, 10)


# To sample only from the actual observations

var3 = CPIVarietyMatchDistribution(v_10, v_23, m, :actual)
var3 = CPIVarietyMatchDistribution(v_10, v_23, m, InflationEvalTools.actual_reweighing)
var3 = CPIVarietyMatchDistribution(v_10, v_23, m, :actual, code_b10, name_10, code_b23, name_23)
var3 = CPIVarietyMatchDistribution(v_10, v_23, m, InflationEvalTools.actual_reweighing, code_b10, name_10, code_b23, name_23)
sample(var3, 10)
