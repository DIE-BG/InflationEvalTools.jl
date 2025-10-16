# Function to generate inflation trajectories without parallel computation

"""
    function gentrajinfl(
            inflfn::F, resamplefn::R, trendfn::T,
            csdata::CountryStructure;
            numreplications = 100,
            rndseed = DEFAULT_SEED,
            showprogress = true
        ) where {F <: InflationFunction, R <: ResampleFunction, T <: TrendFunction}

Computes `numreplications` inflation trajectories using the inflation function
`inflfn::``InflationFunction`, the resampling function
`resamplefn::``TrendFunction` and the specified trend function
`trendfn::``TrendFunction`. The data in the given `CountryStructure` `csdata`
are used.

Unlike the [`pargentrajinfl`](@ref) function, this function does not perform the
computation in a distributed manner.

To achieve reproducibility between different runs of the function, and thus
generate inflation trajectories with different methodologies using the same
resamplings, the generation seed is set according to the iteration number in the
simulation. To control the start of the trajectory generation, the `rndseed`
offset parameter is used, whose default value is the seed
[`DEFAULT_SEED`](@ref).
"""
function gentrajinfl(
        inflfn::F, resamplefn::R, trendfn::T,
        csdata::CountryStructure;
        numreplications = 100,
        rndseed = DEFAULT_SEED,
        showprogress = true
    ) where {F <: InflationFunction, R <: ResampleFunction, T <: TrendFunction}

    # Set up the random number generator
    myrng = MersenneTwister(rndseed)

    # Output cube of trajectories
    periods = infl_periods(csdata)
    n_measures = num_measures(inflfn)
    traj_infl = zeros(Float32, periods, n_measures, numreplications)

    # Progress control
    p = Progress(numreplications; enabled = showprogress)

    # Generate the trajectories
    for k in 1:numreplications
        # Bootstrap sample of the data
        bootsample = resamplefn(csdata, myrng)
        # Application of the trend function
        trended_sample = trendfn(bootsample)

        # Compute the inflation measure
        traj_infl[:, :, k] = inflfn(trended_sample)

        ProgressMeter.next!(p)
    end

    # Return the trajectories
    return traj_infl
end
