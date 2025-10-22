"""
    const LOCAL_RNG = StableRNG(0)
This constant is used to set the random number generator in each
local process, using the `StableRNG` generator with initial seed zero. The
seed will be altered in each iteration of the simulation process. This
guarantees the reproducibility of the results per resampling realization
by choosing the constant [`DEFAULT_SEED`](@ref).
"""
const LOCAL_RNG = StableRNG(0)


## Function to generate simulated inflation trajectories
# This function considers the resampling function as an argument to be able
# to apply different methodologies in the generation of trajectories, as well as the
# trend function to apply

"""
    pargentrayinfl(inflfn::F, resamplefn::R, trendfn::T, csdata::CountryStructure; 
        numreplications = 100, 
        rndseed = DEFAULT_SEED, 
        showprogress = true)

Computes `numreplications inflation trajectories using the inflation function
`inflfn::``InflationFunction`, the resampling function
`resamplefn::``TrendFunction` and the specified trend function
`trendfn::``TrendFunction`. The data in the given `CountryStructure`
`csdata` are used.
tray_infl
Unlike the [`gentrayinfl`](@ref) function, this function implements
distributed computation in processes using `@distributed`. This requires that the
package has been loaded in all compute processes. For example:

```julia 
using Distributed
addprocs(4, exeflags="--project")
@everywhere using HEMI 
```

To achieve reproducibility between different runs of the function, and thus
generate inflation trajectories with different methodologies using the same
resamplings, the generation seed is set according to the iteration number in
the simulation. To control the start of the trajectory generation, the
`rndseed` offset parameter is used, whose default value is the seed
[`DEFAULT_SEED`](@ref).
"""
function pargentrajinfl(
        inflfn::F, resamplefn::R, trendfn::T,
        csdata::CountryStructure;
        numreplications = 100,
        rndseed = DEFAULT_SEED,
        showprogress = true
    ) where {F <: InflationFunction, R <: ResampleFunction, T <: TrendFunction}

    # Output cube of inflation trajectories
    periods = infl_periods(csdata)
    n_measures = num_measures(inflfn)
    traj_infl = SharedArray{eltype(csdata)}(periods, n_measures, numreplications)

    # Variables for progress control
    progress = Progress(numreplications, enabled = showprogress)
    channel = RemoteChannel(() -> Channel{Bool}(numreplications), 1)

    @sync begin
        # Asynchronous task to update progress
        @async while take!(channel)
            next!(progress)
        end

        # Trajectory computation task
        @sync @distributed for k in 1:numreplications
            # Set the seed in the process
            Random.seed!(LOCAL_RNG, rndseed + k)

            # Bootstrap sample of the data
            bootsample = resamplefn(csdata, LOCAL_RNG)
            # Application of the trend function
            trended_sample = trendfn(bootsample)

            # Compute the inflation measure
            traj_infl[:, :, k] = inflfn(trended_sample)

            put!(channel, true)
        end
        put!(channel, false)
    end

    # Return the trajectories
    return sdata(traj_infl)
end
