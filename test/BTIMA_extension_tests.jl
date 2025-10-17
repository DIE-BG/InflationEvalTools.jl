##  ----------------------------------------------------------------------------
#   Test script for the 2025 B-TIMA Extension Methodology
#   ----------------------------------------------------------------------------
using InflationEvalTools
using Test
using CPIDataGT
using Distributions: pdf, Normal
using Statistics: mean, std
using StatsBase: sample

CPIDataGT.load_data()
includet("BTIMA_extension_helpers.jl")  # Extension helper functions

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
var1 = CPIVarietyMatchDistribution(v_10, v_23, m, f)
var1 = CPIVarietyMatchDistribution(v_10, v_23, m, f, code_b10, name_10, code_b23, name_23)

# This gets the mean and the standard deviation
mean(var1)

std(var1)

manual_std = sqrt(sum(var1.weights[j] * (var1.vkdistr[j] .- var1.expected_value)^2 for j in eachindex(var1.vkdistr)) / (sum(var1.weights) - 1))
@test isapprox(std(var1), manual_std; atol=1e-4)

# When created, I can inspect the resulting weights
var1.weights

# I should be able to resample from it 
sample(var1)
sample(var1, 10)

rng = Random.Xoshiro(0)
sample(rng, var1)
sample(rng, var1, 10)

bootstrap_sample = sample(var1, 10000)
@test isapprox(mean(bootstrap_sample), mean(var1); atol=1e-2)

# In-place methods
x = zeros(eltype(var1), 100)
@which sample!(var1, x)

histogram(bootstrap_sample)
barplot(var1.vkdistr, var1.weights)

