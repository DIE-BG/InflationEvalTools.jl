# ResampleSBB.jl - Functions to resample time series with the
# stationary block bootstrap resampling method

# One idea to improve the efficiency of this resampling function could be
# to compute the matrix of monthly averages only once and store it in the
# StationaryBlockBootstrap struct to be applied at simulation time.


# Definition of the SBB resampling function
struct ResampleSBB <: ResampleFunction
    expected_l::Int
    geom_dist::Geometric

    function ResampleSBB(expected_l::Int)
        # Build the geometric distribution with expected value expected_l
        g = Geometric(1 / expected_l)
        new(expected_l, g)
    end
end

# Define which function to use to obtain parametric data.
get_param_function(::ResampleSBB) = param_sbb

# Define the name and tag of the resampling method
method_name(resamplefn::ResampleSBB) = "Stationary block bootstrap with expected block " * string(resamplefn.expected_l) 
method_tag(resamplefn::ResampleSBB) = "SBB-" * string(resamplefn.expected_l)

## Behavior of the Stationary Block Bootstrap resampling function

# Definition of the resampling procedure for matrices with time series in the columns.
# The matrix `vmat` is resampled, generating `numobsresample` new resampled observations in each time series.
# Assumes that numobsresample >= size(vmat, 1)
function (resample_sbb_fn::ResampleSBB)(vmat::AbstractMatrix, numobsresample::Int, rng = Random.GLOBAL_RNG)

    # Obtain resampling indices
    numobs = size(vmat, 1)
    inds = sbb_inds(resample_sbb_fn, numobs, numobsresample, rng)

    # Reorder the rows of the residual matrix to generate the resampling.

    # The residuals are obtained with respect to the averages of each month.
    # avgmat = monthavg(vmat, numobsresample)
    # residmat = vmat - (@view avgmat[1:numobs, :])
    # resamplemat = avgmat + (@view residmat[inds, :])
    
    # The residuals are obtained with respect to the historical average of each basic expenditure.
    avgmat = mean(vmat, dims=1)
    residmat = vmat .- avgmat
    resamplemat = avgmat .+ (@view residmat[inds, :])
    
    resamplemat
end

# Method that resamples the same number of observations (rows) of `vmat` by default.
function (resample_sbb_fn::ResampleSBB)(vmat::AbstractMatrix, rng = Random.GLOBAL_RNG)
    numobsresample = size(vmat, 1)
    resample_sbb_fn(vmat, numobsresample, rng)
end

# Method to obtain resampling indices for a time series of length
# `numobs`. Can be used to exemplify the operation of the Stationary Block Bootstrap
# resampling method
function (resample_sbb_fn::ResampleSBB)(numobs::Int, numobsresample::Int, rng = Random.GLOBAL_RNG)
    sbb_inds(resample_sbb_fn, numobs, numobsresample, rng)
end


# Resampling function for indices with SBB method for a time series of
# length `numobs` and that generates `numobsresample` observations. This code was
# adapted from the DependentBootstrap library:  
# https://github.com/colintbowers/DependentBootstrap.jl/blob/9ff843a09fde9f83983f5af1d863ca65e21fbbec/src/bootinds.jl#L24-L40
function sbb_inds(resample_sbb_fn::ResampleSBB, numobs::Int, numobsresample::Int = numobs, rng = Random.GLOBAL_RNG)
    # IID Bootstrap
    resample_sbb_fn.expected_l <= 1 && return rand(rng, 1:numobs, numobsresample)
    
    # Stationary Block Bootstrap 
    inds = Vector{Int}(undef, numobsresample)
    geom_dist = resample_sbb_fn.geom_dist
    
    (c, geodraw) = (1, 1)
    for n = 1:numobsresample
        # Start a new block
        if c == geodraw 
            inds[n] = rand(rng, 1:numobs)
            geodraw = rand(rng, geom_dist) + 1
            c = 1
        else 
            # Next observation in the existing block
            inds[n-1] == numobs ? (inds[n] = 1) : (inds[n] = inds[n-1] + 1)
            c += 1
        end
    end
    return inds
end

# Obtain average month-to-month variations of the same months of occurrence.
# `numobsresample` observations are resampled from the time series in the
# columns of `vmat`.
function monthavg(vmat, numobsresample = size(vmat, 1))
    # Create the matrix of averages
    cols = size(vmat, 2)
    avgmat = Matrix{eltype(vmat)}(undef, numobsresample, cols)
    
    # Fill the matrix of averages with the averages of each month
    for i in 1:12
        avgmat[i:12:end, :] .= mean(vmat[i:12:end, :], dims=1)
    end
    return avgmat
end