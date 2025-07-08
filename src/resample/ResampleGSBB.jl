# ResampleGSBB.jl - Functions to resample time series with the Generalized Seasonal Block Bootstrap (GSBB) methodology.

# This implementation allows using any block size and maintains the number of observations in the time series.

Base.@kwdef struct ResampleGSBB <: ResampleFunction
    blocklength::Int = 25
    seasonality::Int = 12
end

# Constructor with default monthly seasonality
ResampleGSBB(blocklength::Int) = ResampleGSBB(blocklength, 12)


# Function to obtain parametric data. Unlike the SBB implementation,
# additional methods are implemented for resamplefn that receive as the
# second argument the type Val{:inverse}() so that the dispatch system
# handles obtaining the parametric data with the same resamplefn function,
# since the block size and seasonality data are required.
get_param_function(resamplefn::ResampleGSBB) = cs::CountryStructure -> resamplefn(cs, Val(:inverse))

# Define the name and tag of the resampling method
method_name(resamplefn::ResampleGSBB) = "Seasonal block bootstrap with block size " * string(resamplefn.blocklength) 
method_tag(resamplefn::ResampleGSBB) = "GSBB-" * string(resamplefn.blocklength)


# Definition of the resampling procedure for matrices with time series in the columns.
# The matrix `vmat` is resampled, generating `numobs` new resampled observations in each time series.
function (resamplefn::ResampleGSBB)(vmat::AbstractMatrix, rng = Random.GLOBAL_RNG)
    # Obtain resampling indices
    numobs = size(vmat, 1)
    inds = resamplefn(numobs, rng)

    # Obtain the resampled time series
    resamplemat = vmat[inds, :]
    resamplemat
end


# Function to obtain GSBB resampling indices.
# See algorithm in Dudek, Leśkow, Paparoditis, and Politis (2013)
function (resamplefn::ResampleGSBB)(numobs::Int, rng = Random.GLOBAL_RNG)
    # Algorithm parameters
    T = numobs
    b = resamplefn.blocklength
    d = resamplefn.seasonality

    # Number of blocks to obtain
    l = T ÷ b
    ids = Vector{UnitRange{Int}}(undef, 0)

    for t in 1:b:l*b+1
        R1 = (t - 1) ÷ d
        R2 = (T - b - t) ÷ d

        # Obtain set of possible indices for observation t and sample
        # iid one of these
        St = (t - d*R1):d:(t + d*R2)
        kt = rand(rng, St)
        
        push!(ids, kt:(kt + b - 1))
    end
    # Obtain the resampling indices
    resample_ids = mapreduce(r -> collect(r), vcat, ids)[1:T]
    resample_ids
end


# Function to obtain ranges of resampling indices for computation of parametric data
function (resamplefn::ResampleGSBB)(numobs::Int, ::Val{:inverse})
    # Algorithm parameters
    T = numobs
    b = resamplefn.blocklength
    d = resamplefn.seasonality

    # Number of blocks to obtain
    l = T ÷ b
    ids = Vector{Vector{UnitRange{Int}}}(undef, 0)
    positions = 1:b:l*b+1
    for t in positions
        R1 = (t - 1) ÷ d
        R2 = (T - b - t) ÷ d

        # Obtain set of possible indices for observation t and sample
        # iid one of these
        St = (t - d*R1):d:(t + d*R2)

        # Save the list of possible indices
        if t == last(positions)
            last_block_size = T - t + 1
            push!(ids, [kt:(kt + last_block_size - 1) for kt in St])
        else
            push!(ids, [kt:kt + b - 1 for kt in St])
        end
    end

    ids
end


# Function to obtain parametric VarCPIBase
function (resamplefn::ResampleGSBB)(base::VarCPIBase, ::Val{:inverse})

    # Obtain lists of possible indices for the blocks
    numobs = periods(base)
    ids = resamplefn(numobs, Val(:inverse))

    # Matrix of parametric values (averages)
    vpob = similar(base.v)
    b = resamplefn.blocklength

    # Obtain the averages of the possible blocks
    for p in 1:length(ids)
        # Obtain matrices at the corresponding indices of each block
        block_mats = map(range_ -> base.v[range_, :], ids[p])
        # Obtain averages from the list of indices
        vpob[(b*(p-1) + 1):clamp(b*p, 1:numobs), :] = mean(block_mats)
    end

    # Set up base of average month-to-month variations
    VarCPIBase(vpob, base.w, base.dates, base.baseindex)
end

# Function to obtain parametric CountryStructure
function (resamplefn::ResampleGSBB)(cs::CountryStructure, ::Val{:inverse})
    pob_base = map(base -> resamplefn(base, Val(:inverse)), cs.base)
    getunionalltype(cs)(pob_base)
end