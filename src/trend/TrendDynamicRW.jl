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
struct TrendDynamicRW{T <: AbstractFloat, F <: Function} <: ArrayTrendFunction
    trend::Vector{T}
    L::Int
    phi::T
    sigma::T
    f::F

    function TrendDynamicRW(L::Int, phi::Real, sigma::Real, f::Function; rng = Random.GLOBAL_RNG)
        # Generate valid AR or RW process
        trend = _generate_valid_ar1(L, phi, sigma, f, rng)
        return new{eltype(trend), typeof(f)}(trend, L, phi, sigma, f)
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
        y[t] = phi * y[t - 1] + randn(rng, T) * sigma
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
    return
end

# Name and tag for the dynamic random walk trend function
method_name(trendfn::TrendDynamicRW) = "Dynamic Random Walk Trend (phi=$(trendfn.phi), sigma=$(trendfn.sigma))"
method_tag(trendfn::TrendDynamicRW) = "DRW"

##  ----------------------------------------------------------------------------
#   Helper function to create an array of trend functions with the parameters
#   provided
#   ----------------------------------------------------------------------------

"""
    zeromean_validation(y; atol = 0.1)

Validation function that returns `true` if the mean of the vector `y` is approximately zero,
within the absolute tolerance `atol`.
Vector `y` is typically the generated AR(1) process before exponentiation.
"""
function zeromean_validation(y::AbstractVector; atol = 0.1)
    return isapprox(mean(y), zero(eltype(y)); atol)
end

"""
    create_TrendDynamicRW_array(L::Int = 360, phi::Real = 1.0f0, sigma::Real = 0.05f0, f::Function = zeromean_validation, nfolds::Int = 10; rng = Random.GLOBAL_RNG)

Create a vector of `TrendDynamicRW` objects with the specified parameters.

# Arguments
- `L::Int`: Length of the trend process (default 360).
- `phi::Real`: Autoregressive coefficient (default 1.0).
- `sigma::Real`: Standard deviation of the shocks (default 0.05).
- `f::Function`: Validation function (default `zeromean_validation`).
- `nfolds::Int`: Number of trend functions to generate (default 10).
- `rng`: Random number generator (default `Random.GLOBAL_RNG`).

Returns a `Vector{TrendDynamicRW}`.
"""
function create_TrendDynamicRW_array(;
    L::Int = 360, 
    phi::Real = 1.0f0, 
    sigma::Real = 0.05f0, 
    f::Function = zeromean_validation, 
    nfolds::Int = 10, 
    rndseed = DEFAULT_SEED,
)
    # Set up RNG with seed for reproducibility
    rng = Random.Xoshiro(rndseed) 
    # Create array of TrendDynamicRW objects
    trendfns = [TrendDynamicRW(L, phi, sigma, f; rng) for _ in 1:nfolds]
    return trendfns
end
