##  ----------------------------------------------------------------------------
#   Synthetic Resampler of the B-TIMA Extension Methodology
#
#   We implement a CPI variety match item, called `CPIVarietyMatchDistribution`
#   to hold a sample of old and new observations from the two adjacent CPI
#   datasets. This object is used to sample individual item's monthly price
#   changes as defined in the B-TIMA Extension Methodology document.
#
#   Then, we implement the `ResampleFunction` called `ResampleSynthetic` to
#   implement the resampling procedure to the full CPI datasets represented by
#   `CountryStructure` objects.
#   ----------------------------------------------------------------------------

import Statistics
import StatsBase: StatsBase, ProbabilityWeights, pweights
import Random: AbstractRNG
using UnicodePlots: barplot

##  ----------------------------------------------------------------------------
#   CPIVarietyMatchDistribution
#   ----------------------------------------------------------------------------

"""
    CPIVarietyMatchDistribution(
        prior_dist, 
        actual_dist, 
        month; 
        weighing_function_type=:synthetic, 
        prior_variety_id=nothing, 
        prior_variety_name=nothing, 
        actual_variety_id=nothing, 
        actual_variety_name=nothing
    )

Create a synthetic empirical distribution that matches an item (variety)
between two adjacent CPI bases. The object stores the concatenated
distribution of monthly price changes, sampling weights (according to the
selected reweighing strategy), and metadata such as the month and optional
ids/names.

Arguments
- `prior_dist`, `actual_dist`: vectors of monthly price changes for the
    prior and actual variety samples.
- `month`: integer month (1..12) the observations correspond to.

Keyword arguments
- `weighing_function_type`: `:synthetic` (default), `:prior`, `:actual`, or a
    custom function accepting `(prior_dist, actual_dist)` and returning a
    scalar weighting function `w(x)` used to construct sampling weights.
- optional `prior_variety_id`, `prior_variety_name`, `actual_variety_id`,
    `actual_variety_name` to store identifying metadata.

Example

```julia
# Suppose v_10 and v_23 are two vectors of monthly changes for the same item
# observed in two CPI bases and m is the month (1..12).
var = CPIVarietyMatchDistribution(v_10, v_23, m)
println(var)                     # short display
mean(var)                        # returns the computed expected value
std(var)                         # returns the weighted std dev
StatsBase.sample(var)            # draw one sample (uses stored weights)
StatsBase.sample(var, 10)        # draw 10 samples

# You can explicitly request prior-only or actual-only sampling:
var_prior = CPIVarietyMatchDistribution(v_10, v_23, m, :prior)
var_actual = CPIVarietyMatchDistribution(v_10, v_23, m, :actual)
```

See also: `synthetic_reweighing`, `prior_reweighing`, `actual_reweighing`.
"""
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
            weighing_function_type::Union{Symbol, Function} = synthetic_reweighing,
            prior_variety_id = nothing,
            prior_variety_name = nothing,
            actual_variety_id = nothing,
            actual_variety_name = nothing
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
        weighing_function_dict = Dict(
            :synthetic => synthetic_reweighing,
            :prior => prior_reweighing,
            :actual => actual_reweighing,
        )
        if (weighing_function_type isa Symbol && weighing_function_type in keys(weighing_function_dict))
            weighing_function = weighing_function_dict[weighing_function]
        else
            weighing_function = weighing_function
        end
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

"""
    synthetic_reweighing(prior_dist, actual_dist; a = 0.35, eps = 0.0001)

Create a reweighing function that assigns weights based on how close each value is to 
the mean of the actual distribution, using a normal density function. This is the default
reweighing strategy used in CPIVarietyMatchDistribution.

# Arguments
- `prior_dist`: Vector of monthly price changes from the prior CPI base
- `actual_dist`: Vector of monthly price changes from the actual CPI base
- `a`: Scaling factor for the standard deviation (default: 0.35)
- `eps`: Small value added to avoid zero variance (default: 0.0001)

Returns a function `w(x)` that computes weights proportional to the normal density 
of the absolute difference between x and the actual mean.
"""
function synthetic_reweighing(prior_dist, actual_dist; a = 0.35, eps = 0.0001)
    vdistr = [prior_dist..., actual_dist...]
    # Computes the mean of the new observations
    actual_mean = mean(actual_dist)
    density = Normal(0, a * std(vdistr) + eps)
    return x -> pdf(density, abs(x - actual_mean))
end

"""
    prior_reweighing(prior_dist, actual_dist)

Create a reweighing function that only samples from the prior distribution, assigning
unit weights to values from the prior distribution and zero weights to values from
the actual distribution.

# Arguments
- `prior_dist`: Vector of monthly price changes from the prior CPI base
- `actual_dist`: Vector of monthly price changes from the actual CPI base (unused)

Returns a function `w(x)` that assigns 1 if x is in the prior distribution and 0 otherwise.
Used when `:prior` is specified as the weighing_function_type in CPIVarietyMatchDistribution.
"""
function prior_reweighing(prior_dist, actual_dist)
    T = eltype(prior_dist)
    reweighing_function = x -> x in prior_dist ? one(T) : zero(T)
    return reweighing_function
end

"""
    actual_reweighing(prior_dist, actual_dist)

Create a reweighing function that only samples from the actual distribution, assigning
unit weights to values from the actual distribution and zero weights to values from
the prior distribution.

# Arguments
- `prior_dist`: Vector of monthly price changes from the prior CPI base (unused)
- `actual_dist`: Vector of monthly price changes from the actual CPI base

Returns a function `w(x)` that assigns 1 if x is in the actual distribution and 0 otherwise.
Used when `:actual` is specified as the weighing_function_type in CPIVarietyMatchDistribution.
"""
function actual_reweighing(prior_dist, actual_dist)
    T = eltype(actual_dist)
    reweighing_function = x -> x in actual_dist ? one(T) : zero(T)
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

    show(io, MIME("text/plain"), p)
    return
end

function Base.show(io::IO, cpi_match_dist::CPIVarietyMatchDistribution{T}) where {T}
    m = cpi_match_dist.month
    JH = length(cpi_match_dist)
    expected_value = round(mean(cpi_match_dist), digits = 4)
    print(io, "CPIVMD")
    print(io, "($m, $(JH), $expected_value)")
    return
end


# Basic stats methods
Statistics.mean(cpi_match_dist::CPIVarietyMatchDistribution) = cpi_match_dist.expected_value
Statistics.std(cpi_match_dist::CPIVarietyMatchDistribution) = StatsBase.std(
    cpi_match_dist.vkdistr, cpi_match_dist.weights;
    mean = cpi_match_dist.expected_value, corrected = true
)


# Extend sampling methods
StatsBase.sample(cpi_match_dist::CPIVarietyMatchDistribution) =
    StatsBase.sample(Random.default_rng(), cpi_match_dist.vkdistr, cpi_match_dist.weights)

StatsBase.sample(cpi_match_dist::CPIVarietyMatchDistribution, n::Int) =
    StatsBase.sample(Random.default_rng(), cpi_match_dist.vkdistr, cpi_match_dist.weights, n)

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
) = StatsBase.sample!(Random.default_rng(), cpi_match_dist.vkdistr, cpi_match_dist.weights, x; replace, ordered)


##  ----------------------------------------------------------------------------
#   ResampleSynthetic: Resampling function using an array of the
#   CPIVarietyMatchDistribution objects
#   ----------------------------------------------------------------------------

"""
    ResampleSynthetic(
        base::VarCPIBase, 
        matching_structure::Matrix{CPIVarietyMatchDistribution}, 
        numperiods = periods(base)
    )

Create a resampling function that implements the B-TIMA Extension Methodology
for generating synthetic samples from matched CPI items between two adjacent CPI
bases. Importantly, `ResampleSynthetic` is to be used only for `VarCPIBase`
objects because the required matching structure is specific to the `VarCPIBase`.

# Arguments
- `base`: A VarCPIBase object that provides reference dimensions and dates
- `matching_structure`: A 2D array of CPIVarietyMatchDistribution objects with
  shape (periods, items) containing the matching distributions for each item and
  month
- `numperiods`: Optional number of periods to generate in the resampled series
  (defaults to the number of periods in base)

# Example
```julia
# Create matching distributions for each item and month
matching_array = Array{CPIVarietyMatchDistribution}(undef, 12, nitems)
for j in 1:nitems, m in 1:12
    v_prior, v_actual = get_obs_month(prior_codes[j], actual_codes[j], m)
    # Use synthetic reweighing for items 1-99
    # prior-only for items 100-349
    # actual-only for remaining items
    weighing = j < 100 ? :synthetic : (j < 350 ? :prior : :actual)
    matching_array[m,j] = CPIVarietyMatchDistribution(v_prior, v_actual, m, weighing)
end

# Create resampler with 5 years of data
synth_resampler = ResampleSynthetic(base, matching_array, 12*5)

# Generate a synthetic sample
varbase_sample = synth_resampler(base)

# Get the population mean parameters
population_fn = get_param_function(synth_resampler)
population_base = population_fn(base)
```

See also: [`CPIVarietyMatchDistribution`](@ref), [`synthetic_reweighing`](@ref),
[`prior_reweighing`](@ref), [`actual_reweighing`](@ref)
"""
struct ResampleSynthetic{A} <: ResampleFunction
    matching::A
    periods::Int

    function ResampleSynthetic(base::VarCPIBase, matching_structure::A, numperiods::Int = periods(base)) where {A}
        _validate_synthetic_params(base, matching_structure, numperiods)
        return new{A}(matching_structure, numperiods)
    end

end

method_name(::ResampleSynthetic) = "IID Bootstrap Synthetic Resampling"
method_tag(::ResampleSynthetic) = "SYNTH"

function _validate_synthetic_params(base::VarCPIBase, matching_structure, numperiods)
    # Check for the dimension of the matching structure
    ndims(matching_structure) == 2 || error("Matching structure requires a 2-dimensional object with (periods, items)")
    # Check at least the same number of months in the data as in the matching structure
    base_periods = periods(base)
    matching_periods = size(matching_structure, 1)
    if (matching_periods < 12 && matching_periods < base_periods)
        error("Not enough matching periods for the specified base")
    elseif (matching_periods > 12)
        @warn "More than 12 periods specified in the matching structure"
    end
    # Check all matching objects have the same months along the same rows
    are_same_months_axes = mapreduce(==, eachrow(matching_structure)) do row
        # Get months of all matching objects
        months = [cpi_match_dist.month for cpi_match_dist in row]
        return allequal(months)
    end
    are_same_months_axes || error("Matching structure contains mixed period numbers in its rows")

    # Check numperiods is positive
    (numperiods > 0) || error("Resampling periods must be positive")

    # Perform checks on appropriate sizes for resampling
    base_nitems = size(base.v, 2)
    sampler_nitems = size(matching_structure, 2)
    (sampler_nitems == base_nitems) || error("The matching structure has less items than the VarCPIBase object")

    # Check start month of the VarCPIbase coincides with months in the matching structure
    base_initial_month = month(first(base.dates))
    sampler_initial_month = first(matching_structure).month
    return (sampler_initial_month == base_initial_month) || error("Matching objects month order differs from the VarCPIBase dates")
end

## Definition of its resampling procedures

# This type is not conceived to be used for general resampling of CountryStructures
function (::ResampleSynthetic)(::CountryStructure, ::AbstractRNG)
    error("This bootstrap resampling function does not operate with `CountryStructures` because of the required matching structure")
end

# The general definition for VarCPIBases is good enough for this resample function
# function (resamplefn::ResampleFunction)(base::VarCPIBase, rng = Random.GLOBAL_RNG)

# We redefine how to resample from a matrix of monthly price changes (periods, items)
function (resamplefn::ResampleSynthetic)(vmat::AbstractMatrix, rng::AbstractRNG = Random.GLOBAL_RNG)
    # Take resampling periods from original matrix or from the attribute of the
    # resample function
    nperiods = (resamplefn.periods === nothing) ? size(vmat, 1) : resamplefn.periods
    nitems = size(vmat, 2)
    # Create and return the resampled series
    resampled_vmat = similar(vmat, nperiods, nitems)

    # Sample from the CPI matching items stored in the resample function
    for j in 1:nitems, m in 1:min(nperiods, 12)
        # Get the CPI matching object for item j and month m
        cpi_match_dist = resamplefn.matching[m, j]
        # Sample from the distribution to fill observations from the same months
        resampled_slice = @view resampled_vmat[m:12:end, j]
        StatsBase.sample!(rng, cpi_match_dist, resampled_slice)
    end
    return resampled_vmat
end


## Population dataset function

function get_param_function(resamplefn::ResampleSynthetic)
    matching_structure = resamplefn.matching
    numperiods = resamplefn.periods
    # Build a closure to return the appropriate size VarCPIBase objects with the
    # population data
    population_data_fn = base -> param_synth_sampler_fn(base, matching_structure, numperiods)
    return population_data_fn
end

function param_synth_sampler_fn(base::VarCPIBase, matching_structure, numperiods)
    # Get the population data from the array of matching objects
    population_vmat_matching = map(mean, matching_structure)

    # Form a population data matrix with the same number of periods as the resample function
    repsize = div(numperiods, size(matching_structure, 1)) + 1
    population_vmat = repeat(population_vmat_matching, repsize)
    population_vmat = population_vmat[1:numperiods, :]

    # Construct a VarCPIBase object and return it
    start_date = first(base.dates)
    dates = start_date:Month(1):(start_date + Month(numperiods - 1))
    return VarCPIBase(population_vmat, base.w, dates, base.baseindex)
end
