##  ----------------------------------------------------------------------------
#   Synthetic Resampler of the B-TIMA Extension Methodology
#
#   We implement a CPI variety match item, called `CPIVarietyMatch` to hold a
#   sample of old and new observations from the two adjacent CPI datasets. This
#   object is used to sample individual item's monthly price changes as defined
#   in the B-TIMA Extension Methodology document.
#
#   Then, we implement the `ResampleFunction` called `ResampleSynthetic` to
#   implement the resampling procedure to the full CPI datasets represented by
#   `CountryStructure` objects.
#   ----------------------------------------------------------------------------

import Statistics
import StatsBase: FrequencyWeights, fweights
import Random: AbstractRNG

struct CPIVarietyMatchDistribution{T <: AbstractFloat}
    vkdistr::Vector{T}          # Distribution of variety k of monthly price changes
    weights::FrequencyWeights   # Frequency weights vector for resampling
    J::Int      # Number of observations in the prior empirical distribution
    expected_value::T
    month::Int  # Month number
    prior_variety_id::Union{AbstractString, Nothing}
    prior_variety_name::Union{AbstractString, Nothing}
    obs_variety_id::Union{AbstractString, Nothing}
    obs_variety_name::Union{AbstractString, Nothing}

    function CPIVarietyMatchDistribution(
            prior_dist::AbstractVector{T},
            obs_dist::AbstractVector{T},
            month::Int,
            weighing_function::Function,
            prior_variety_id = nothing,
            prior_variety_name = nothing,
            obs_variety_id = nothing,
            obs_variety_name = nothing
        ) where {T <: AbstractFloat}

        # Validation
        (isempty(prior_dist) || isempty(obs_dist)) && error("Prior or observed distributions are empty")
        (1 <= month <= 12) || error("Incorrect month specified")

        # Mean of the observed distribution
        vkdistr = vcat(prior_dist, obs_dist)
        J = length(prior_dist)
        H = length(obs_dist)
        # Compute sampling weights according to weighing function
        g = weighing_function(vkdistr)
        wkj = _reweigh(vkdistr, g, H)
        # Compute the expected value
        expected_value = sum(vkdistr .* wkj) / sum(wkj)

        return new{T}(
            vkdistr, wkj, J, expected_value, month,
            prior_variety_id,
            prior_variety_name,
            obs_variety_id,
            obs_variety_name,
        )
    end
end


# Returns the normalized weights
function _reweigh(vkdistr::AbstractArray{T}, f::Function, H::Int) where {T}
    # Computes the mean of the new observations
    vk_new_mean = mean(last(vkdistr, H))
    # Computes the reweighing factor
    wkj = @. f(abs(vkdistr - vk_new_mean))
    # Normalize the weights
    # Wk = sum(wkj)
    # wkj_norm = wkj / Wk
    weights = convert.(T, wkj)
    return fweights(weights)
end

# Base methods
Base.length(cpi_match_dist::CPIVarietyMatchDistribution) = length(cpi_match_dist.vkdistr)
Base.eltype(::CPIVarietyMatchDistribution{T}) where {T} = T

function Base.show(io::IO, ::MIME"text/plain", cpi_match_dist::CPIVarietyMatchDistribution)
    vkdistr = cpi_match_dist.vkdistr
    J = cpi_match_dist.J
    H = length(cpi_match_dist) - J
    prior_mean = mean(first(vkdistr, J))
    obs_mean = mean(last(vkdistr, H))
    print(io, typeof(cpi_match_dist))
    print(io, " prior: $J mean: $(prior_mean)")
    print(io, " obs: $H mean: $(obs_mean)")
    print(io, "\n some more info...")

    # println(io, "Prior variety: $(x.prior_variety_id !== nothing ? x.prior_variety_id : "unnamed") (n=$(x.J))")
    # println(io, "Observed variety: $(x.obs_variety_id !== nothing ? x.obs_variety_id : "unnamed") (n=$(length(x) - x.J))")
    # print(io, "Month: $(x.month), Mean: $(round(mean(x), digits=4)), Std: $(round(std(x), digits=4))")
end

function Base.show(io::IO, cpi_match_dist::CPIVarietyMatchDistribution)
    vkdistr = cpi_match_dist.vkdistr
    J = cpi_match_dist.J
    H = length(cpi_match_dist) - J
    expected_value = mean(cpi_match_dist)
    print(io, typeof(cpi_match_dist))
    print(io, "($J, $H, $expected_value)")
    # print(io, " obs: $H mean: $(obs_mean)")
end

# Extend sampling methods
StatsBase.sample(cpi_match_dist::CPIVarietyMatchDistribution) =
    StatsBase.sample(StatsBase.default_rng(), cpi_match_dist.vkdistr, cpi_match_dist.weights)

StatsBase.sample(cpi_match_dist::CPIVarietyMatchDistribution, n::Int) =
    StatsBase.sample(StatsBase.default_rng(), cpi_match_dist.vkdistr, cpi_match_dist.weights, n)

StatsBase.sample(rng::AbstractRNG, cpi_match_dist::CPIVarietyMatchDistribution) =
    StatsBase.sample(rng, cpi_match_dist.vkdistr, cpi_match_dist.weights)

StatsBase.sample(rng::AbstractRNG, cpi_match_dist::CPIVarietyMatchDistribution, n::Int) =
    StatsBase.sample(rng, cpi_match_dist.vkdistr, cpi_match_dist.weights, n)

StatsBase.sample!(
    rng::AbstractRNG, cpi_match_dist::CPIVarietyMatchDistribution, x;
    replace = true, ordered = false
) = StatsBase.sample!(rng, cpi_match_dist.vkdistr, cpi_match_dist.weights, x; replace, ordered)

StatsBase.sample!(
    cpi_match_dist::CPIVarietyMatchDistribution, x;
    replace = true, ordered = false
) = StatsBase.sample!(StatsBase.default_rng(), cpi_match_dist.vkdistr, cpi_match_dist.weights, x; replace, ordered)

# Basic stats methods
Statistics.mean(cpi_match_dist::CPIVarietyMatchDistribution) =
    cpi_match_dist.expected_value
Statistics.std(cpi_match_dist::CPIVarietyMatchDistribution) =
    StatsBase.std(
    cpi_match_dist.vkdistr, cpi_match_dist.weights;
    mean = cpi_match_dist.expected_value, corrected = true
)
