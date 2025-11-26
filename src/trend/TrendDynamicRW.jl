## Definition of type for dynamic random walk trend

import Random

"""
    TrendDynamicRW{T, F<:Function} <: ArrayTrendFunction

Type to represent a dynamic random walk (or AR(1)) trend function.
Generates an AR(1) process that satisfies a validation function.

# Fields
- `trend::Vector{T}`: The generated trend vector.
- `L::Int`: Length of the process.
- `phi::T`: Autocorrelation parameter.
- `sigma::T`: Standard deviation of shocks.
- `f::F`: Validation function to select realizations of the process.

# Constructor
    TrendDynamicRW(L::Int, phi::Real, sigma::Real, f::Function; rng=Random.GLOBAL_RNG)

Creates a `TrendDynamicRW` object by generating an AR(1) process with parameters
`L`, `phi`, and `sigma` until the validation function `f` returns `true`.
"""
struct TrendDynamicRW{T<:AbstractFloat, F<:Function} <: ArrayTrendFunction
    trend::Vector{T}
    L::Int
    phi::T
    sigma::T
    f::F

    function TrendDynamicRW(L::Int, phi::Real, sigma::Real, f::Function; rng=Random.GLOBAL_RNG)
        # Generate valid AR or RW process
        trend = _generate_valid_ar1(L, phi, sigma, f, rng)
        new{eltype(trend), typeof(f)}(trend, L, phi, sigma, f)
    end
end

# Private function to generate AR(1) process
function _generate_ar1(L::Int, phi::Real, sigma::Real, rng::Random.AbstractRNG)
    T = promote_type(typeof(phi), typeof(sigma))
    y = Vector{T}(undef, L)
    # Initialize first value only with noise
    y[1] = randn(rng, T) * sigma
    # Generate AR(1) process
    for t in 2:L
        y[t] = phi * y[t-1] + randn(rng, T) * sigma
    end
    return y
end

# Private function to generate valid AR(1) process
function _generate_valid_ar1(L::Int, phi::Real, sigma::Real, f::Function, rng::Random.AbstractRNG)
    # Repeat until validation function is satisfied
    while true
        y = _generate_ar1(L, phi, sigma, rng)
        # Check validation function and return if process is valid
        # Note it returns the exponentiated values to get the trend factor
        f(y) && return exp.(y)
    end
end

# Name and tag for the dynamic random walk trend function
method_name(trendfn::TrendDynamicRW) = "Dynamic Random Walk Trend (phi=$(trendfn.phi), sigma=$(trendfn.sigma))"
method_tag(trendfn::TrendDynamicRW) = "DRW"
