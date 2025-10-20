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
import StatsBase: ProbabilityWeights, pweights
import Random: AbstractRNG
using UnicodePlots: barplot

Base.@kwdef struct CPIVarietyMatchDistribution{T <: AbstractFloat}
    vkdistr::Vector{T}              # Distribution of variety k of monthly price changes
    weights::ProbabilityWeights     # Frequency weights vector for resampling
    actual_mask::BitVector          # Mask for actual observations
    expected_value::T               # Synthetic distribution expected value
    month::Int                      # Month number
    prior_variety_id::Union{AbstractString, Nothing}
    prior_variety_name::Union{AbstractString, Nothing}
    actual_variety_id::Union{AbstractString, Nothing}
    actual_variety_name::Union{AbstractString, Nothing}

    function CPIVarietyMatchDistribution(
            prior_dist::AbstractVector{T},
            actual_dist::AbstractVector{T},
            month::Int,
            weighing_function::Function = synthetic_reweighing,
            prior_variety_id= nothing,
            prior_variety_name= nothing,
            actual_variety_id= nothing,
            actual_variety_name= nothing
        ) where {T <: AbstractFloat}

        # Validation
        (isempty(prior_dist) || isempty(actual_dist)) && error("Prior or observed distributions are empty")
        (1 <= month <= 12) || error("Incorrect month specified")

        # Mean of the observed distribution
        vkdistr = vcat(prior_dist, actual_dist)
        J = length(prior_dist)
        H = length(actual_dist)
        mask = [falses(J); trues(H)]

        # Reorder prior and new obs
        sortinds = sortperm(vkdistr)
        vkdistr = vkdistr[sortinds]
        actual_mask = mask[sortinds]

        # Compute sampling weights according to weighing function
        g = weighing_function(prior_dist, actual_dist)
        wkj = _reweigh(vkdistr, g)

        # Compute the expected value
        expected_value = sum(vkdistr .* wkj) / sum(wkj)

        return new{T}(
            vkdistr, wkj, actual_mask, expected_value, month,
            prior_variety_id,
            prior_variety_name,
            actual_variety_id,
            actual_variety_name,
        )
    end
end

# const CPIVMD{T} = CPIVarietyMatchDistribution{T}

function CPIVarietyMatchDistribution(
        prior_dist::AbstractVector{T},
        actual_dist::AbstractVector{T},
        month::Int,
        weighing_function_type::Symbol = :synthetic,
        prior_variety_id= nothing,
        prior_variety_name= nothing,
        actual_variety_id= nothing,
        actual_variety_name= nothing
    ) where {T <: AbstractFloat}

    weighing_function_dict = Dict(
        :synthetic => synthetic_reweighing,
        :prior => prior_reweighing,
        :actual => actual_reweighing,
    )

    return CPIVarietyMatchDistribution(
        prior_dist,
        actual_dist,
        month,
        weighing_function_dict[weighing_function_type],
        prior_variety_id,
        prior_variety_name,
        actual_variety_id,
        actual_variety_name,
    )
end

# Returns the normalized weights
function _reweigh(vkdistr::AbstractArray{T}, f::Function) where {T}
    # Apply weighing function to individual monthly price changes
    wkj = @. f(vkdistr)
    # Normalize the weights
    Wk = sum(wkj)
    wkj_norm = wkj / Wk
    # Convert the weights to the same type as the synthetic distribution
    weights = convert.(T, wkj_norm)
    return pweights(weights)
end

function synthetic_reweighing(prior_dist, actual_dist; a = 0.35, eps=0.0001)
    vdistr = [prior_dist..., actual_dist...]
    # Computes the mean of the new observations
    actual_mean = mean(actual_dist)
    density = Normal(0, a * std(vdistr) + eps)
    return x -> pdf(density, abs(x - actual_mean))
end

function prior_reweighing(prior_dist, actual_dist)
    T = eltype(prior_dist)
    reweighing_function = x -> x in prior_dist ? one(T) : zero(T)
    return reweighing_function
end

function actual_reweighing(prior_dist, actual_dist)
    T = eltype(actual_dist)
    reweighing_function =  x -> x in actual_dist ? one(T) : zero(T)
    return reweighing_function
end



# Base methods
Base.length(cpi_match_dist::CPIVarietyMatchDistribution) = length(cpi_match_dist.vkdistr)
Base.eltype(::CPIVarietyMatchDistribution{T}) where {T} = T

function Base.show(io::IO, ::MIME"text/plain", cpi_match_dist::CPIVarietyMatchDistribution)
    vkdistr = cpi_match_dist.vkdistr
    actual_mask = cpi_match_dist.actual_mask
    H = sum(cpi_match_dist.actual_mask)
    J = length(cpi_match_dist) - H
    prior_mean = mean(vkdistr[.!actual_mask])
    actual_mean = mean(vkdistr[actual_mask])
    prior_info = string(cpi_match_dist.prior_variety_id) * ":" * string(cpi_match_dist.prior_variety_name)
    actual_info = string(cpi_match_dist.actual_variety_id) * ":" * string(cpi_match_dist.actual_variety_name)
    print(io, "$(J + H)-element ", typeof(cpi_match_dist), " for month:$(cpi_match_dist.month), prior:$J, actual:$H\n")
    print(io, "↳ Prior mean $(prior_mean) of $(prior_info)\n")
    print(io, "↳ Actual mean $(actual_mean) of $(actual_info)\n")

    # Draw a barplot to show the synthetic distribution
    p = barplot(
        cpi_match_dist.vkdistr,
        cpi_match_dist.weights,
        color = [is_new ? :blue : :green for is_new in cpi_match_dist.actual_mask],
        xlabel = "Synthetic empirical distribution",
        # ylabel = "Monthly price changes"
    )

    return show(io, MIME("text/plain"), p)
end

function Base.show(io::IO, cpi_match_dist::CPIVarietyMatchDistribution{T}) where {T}
    m = cpi_match_dist.month
    H = sum(cpi_match_dist.actual_mask)
    J = length(cpi_match_dist) - H
    expected_value = mean(cpi_match_dist)
    print(io, typeof(cpi_match_dist))
    return print(io, "($m, $J, $H, $expected_value)")
end


# Basic stats methods
Statistics.mean(cpi_match_dist::CPIVarietyMatchDistribution) = cpi_match_dist.expected_value
Statistics.std(cpi_match_dist::CPIVarietyMatchDistribution) = StatsBase.std(
    cpi_match_dist.vkdistr, cpi_match_dist.weights;
    mean = cpi_match_dist.expected_value, corrected = true
)


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
