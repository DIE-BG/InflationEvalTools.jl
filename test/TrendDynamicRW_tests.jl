## Test script for TrendDynamicRW
using InflationEvalTools
using Test
using Statistics
using Random
using CPIDataGT
    
# Test 1: Basic instantiation and properties
L = 100
phi = 0.5
sigma = 1.0
f_always_true = x -> true

rng = Random.MersenneTwister(123)
trend_obj = TrendDynamicRW(L, phi, sigma, f_always_true; rng=rng)

@test trend_obj isa TrendDynamicRW
@test length(trend_obj.trend) == L
@test trend_obj.L == L
@test trend_obj.phi == phi
@test trend_obj.sigma == sigma
@test trend_obj.f == f_always_true
@test method_tag(trend_obj) == "DRW"
@test occursin("Dynamic Random Walk Trend", method_name(trend_obj))

# Test 2: Validation function logic
# Define a validation function that requires the mean to be positive
f_positive_mean = x -> mean(x) > 0.1

# It might take a few tries to generate one, so we use a loop internally in the constructor
trend_pos = TrendDynamicRW(L, phi, sigma, f_positive_mean)
@test mean(trend_pos.trend) > 1

# Define a validation function that requires the last value to be negative
f_last_neg = x -> x[end] < -0.5
trend_neg = TrendDynamicRW(L, phi, sigma, f_last_neg)
@test trend_neg.trend[end] < 1

# Test 3: AR(1) properties (statistical check)
# With a large L, the sample autocorrelation at lag 1 should be close to phi
L_large = 10000
phi_target = 0.7
sigma_target = 0.5
trend_stat = TrendDynamicRW(L_large, phi_target, sigma_target, f_always_true)

# Calculate lag-1 autocorrelation, here we use the log of the trend since it's an exponentiated AR(1)
y = log.(trend_stat.trend)
y_mean = mean(y)
numerator = sum((y[2:end] .- y_mean) .* (y[1:end-1] .- y_mean))
denominator = sum((y .- y_mean).^2)
# Approximate autocorrelation, ignoring end effects for denominator
rho_1 = numerator / sum((y[1:end-1] .- mean(y[1:end-1])).^2) 

# Check if lag-1 coefficient is close (allow some tolerance)
@test isapprox(rho_1, phi_target, atol=0.05)

# Check unconditional variance variance: Var(y) = sigma^2 / (1 - phi^2)
expected_var = sigma_target^2 / (1 - phi_target^2)
@test isapprox(var(y), expected_var, atol=0.2)

# Test 4: Application to CountryStructure (using dummy data if needed, or just checking type hierarchy)
# Since TrendDynamicRW <: ArrayTrendFunction, it should work with the existing machinery.
# We can check if it is indeed an ArrayTrendFunction
@test trend_obj isa InflationEvalTools.ArrayTrendFunction


# Test 5: Zero-mean around specific periods
function f_zeromean(y)
    # Check if mean over entire series is close to zero
    return abs(mean(y[121:145])) < 0.05 
end

# It might take a few tries to generate one, so we use a loop internally in the constructor
trend_zeromean = TrendDynamicRW(150, 0.9, 0.50, f_zeromean)

y = log.(trend_zeromean.trend[121:145])
@test abs(mean(y)) < 0.05


## Test 6: Application to VarCPIBase
# All-negative VarCPIBase followed by all-positive VarCPIBase

# All-negative VarCPIBase
dates1 = collect(Date(2001,1):Month(1):Date(2010,12))
v1 = -1 * ones(Float32, length(dates1), 218)
w1 = ones(Float32, 218) / 218 # Equal weights

# All-positive VarCPIBase
dates2 = collect(Date(2011,1):Month(1):Date(2023,12))
v2 = 1 * ones(Float32, length(dates2), 279)
w2 = ones(Float32, 279) / 279

# Create CountryStructure
base1 = VarCPIBase(v1, w1, dates1, 100.0f0)
base2 = VarCPIBase(v2, w2, dates2, 100.0f0)
cs = UniformCountryStructure(base1, base2)

# Zero mean around the last 40 periods
function f_zeromean_last40(y)
    # Check if mean over entire series is close to zero
    return abs(mean(y[end-39:end])) < 0.025
end

# Create trend function with zero-mean constraint
trend_zeromean = TrendDynamicRW(
    length(dates1)+length(dates2), 
    0.9f0, 
    0.05f0, 
    f_zeromean_last40,
)

# Apply the trend function to the CountryStructure
trended_cs = trend_zeromean(cs)

# First base should be unchanged (all negative changes)
@test trended_cs.base[1].v == cs.base[1].v
# Second base should be modified
@test trended_cs.base[2].v != cs.base[2].v
# Check that the trend was applied correctly to all the positive monthly price changes
@test trended_cs.base[2].v[:, 1] ≈ trend_zeromean.trend[length(dates1)+1:length(dates1)+length(dates2)]


## Test 7: Reproducibility with RNG
# Create two trends with the same RNG seed and check they are identical

rng1 = Xoshiro(314159)
trend1 = TrendDynamicRW(100, 0.8, 0.2, f_always_true; rng=rng1)

rng2 = Xoshiro(314159)
trend2 = TrendDynamicRW(100, 0.8, 0.2, f_always_true; rng=rng2)

@test trend1.trend == trend2.trend
