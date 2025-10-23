## Resampling of CPIDataBase objects

# Definition of the resampling function by month of occurrence
"""
    ResampleExtendedSVM <: ResampleFunction

Define a resampling function to resample the time series for the same months of
occurrence. Sampling is performed independently for each time series in the
columns of the matrix of monthly price changes. 

# Example
In this example, `GTDATA23` is a `CountryStructure` and contains three
`VarCPIBase` objects. For each CPI dataset, the resampling is performed with the
same length as the vector used as a parameter in ResampleExtendedSVM().

```julia
resamplefn = ResampleExtendedSVM([150, 180, 50])
resamplefn(GTDATA23)
```
"""
struct ResampleExtendedSVM <: ResampleFunction
    extension_periods::Union{Int, Vector{Int}}
end

# Define how to resample matrices with time series in the columns.
# Uses the internal function `scramblevar`.
function (resamplefn::ResampleExtendedSVM)(cs::CountryStructure, rng = Random.GLOBAL_RNG)

    if isa(resamplefn.extension_periods, Vector) &&
            length(resamplefn.extension_periods) != length(cs.base)
        error("The vector of periods must have the same number of VarCPIBase objects as the CountryStructure")
    end

    # If a single integer is provided, use the same for resampling all the
    # VarCPIBase objects
    if length(resamplefn.extension_periods) == 1
        periods = fill(resamplefn.extension_periods, length(cs.base))
    else
        periods = resamplefn.extension_periods
    end

    # Obtain resampled bases
    ps = (periods...,)
    base_boot = map((b, extension_periods) -> resamplefn(b, extension_periods, rng), cs.base, ps)
    # Return new CountryStructure
    return typeof(cs)(base_boot)

end

function (resamplefn::ResampleExtendedSVM)(base::VarCPIBase, extension_periods::Int = periods(base), rng = Random.GLOBAL_RNG)
    # Resample the matrix of monthly price changes of the VarCPIBase
    v_boot = resamplefn(base.v, extension_periods, rng)
    # Extend the dates
    startdate = base.dates[1]
    dates = startdate:Month(1):Date(startdate + Month(extension_periods - 1))

    return VarCPIBase(v_boot, base.w, dates, base.baseindex)
end

function (resamplefn::ResampleExtendedSVM)(vmat::AbstractMatrix, extension_periods::Int, rng = Random.GLOBAL_RNG)
    return scramblevar(vmat, rng, extension_periods)
end

# Define the name and tag of the resampling method
method_name(resamplefn::ResampleExtendedSVM) = "Extended IID bootstrap by months of occurrence"
method_tag(resamplefn::ResampleExtendedSVM) = "ESVM"

## Population dataset function 

# Similar to param_scramblevar_fn, but with extension for dates
function param_scramblevar_ext_fn(base::VarCPIBase, sample_periods::Int = periods(base))
    # Get the matrix of average monthly price changes
    month_mat = monthavg(base.v, sample_periods)
    # Form the VarCPIbase with monthly averages with appropriate dates
    start_date = first(base.dates)
    dates = start_date:Month(1):(start_date + Month(sample_periods - 1))
    return VarCPIBase(month_mat, base.w, dates, base.baseindex)
end

function get_param_function(resamplefn::ResampleExtendedSVM)
    # Define a function with the sample periods embedded to obtain the
    # population CPI datasets
    function param_scramblevar_ext_closure_fn(cs::CountryStructure)
        num_bases = length(cs.base)
        num_ext_periods = length(resamplefn.extension_periods)
        # Check appropriate sizes of the number of sampling periods vs. number
        # of VarCPIBase objects
        if num_ext_periods == 1
            sample_periods = fill(resamplefn.extension_periods, num_bases)
        elseif num_ext_periods == num_bases
            sample_periods = resamplefn.extension_periods
        else
            error("CountryStructure has a different number of VarCPIBases ($num_bases) than the resample function ($num_ext_periods)")
        end
        # Compute the population datasets using the number of sample periods
        # specified in the sampler
        population_varbases = map(param_scramblevar_ext_fn, cs.base, sample_periods)
        return getunionalltype(cs)(population_varbases...)
    end

    return param_scramblevar_ext_closure_fn
end
