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
function (resamplefn::ResampleExtendedSVM)(cs::CountryStructure, rng::AbstractRNG = Random.GLOBAL_RNG)

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
    bootstrap_varbases = map((b, extension_periods) -> resamplefn(b, rng, extension_periods), cs.base, ps)

    # Return new CountryStructure
    cstype = getunionalltype(cs)
    bootstrap_cs = cstype(bootstrap_varbases...)
    bootstrap_cs = _fix_countrystructure_dates(bootstrap_cs)
    return bootstrap_cs

end

function (resamplefn::ResampleExtendedSVM)(base::VarCPIBase, rng::AbstractRNG = Random.GLOBAL_RNG, extension_periods::Int = first(resamplefn.extension_periods))
    # Resample the matrix of monthly price changes of the VarCPIBase
    v_boot = resamplefn(base.v, rng, extension_periods)
    # Extend the dates
    startdate = base.dates[1]
    dates = startdate:Month(1):Date(startdate + Month(extension_periods - 1))

    return VarCPIBase(v_boot, base.w, dates, base.baseindex)
end

function (resamplefn::ResampleExtendedSVM)(vmat::AbstractMatrix, rng::AbstractRNG = Random.GLOBAL_RNG, extension_periods::Int = first(resamplefn.extension_periods))
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
    # population CPI datasets from a CountryStructure 
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
        cstype = getunionalltype(cs)
        population_cs = cstype(population_varbases...)
        population_cs = _fix_countrystructure_dates(population_cs)
        return population_cs
    end

    # Population data for a VarCPIBase. Added for completeness, when calling a
    # ResampleExtendedSVM within a ResampleMixture, the ResampleMixture will
    # call this function
    function param_scramblevar_ext_closure_fn(base::VarCPIBase)
        # If the first entry of the extension_periods when asked for the population data of a VarCPIBase
        sample_periods = first(resamplefn.extension_periods)
        return param_scramblevar_ext_fn(base, sample_periods)
    end

    return param_scramblevar_ext_closure_fn
end


## Helper to fix a CountryStructure's VarCPIBase dates
function _fix_countrystructure_dates(cs::CountryStructure)
    nperiods = periods(cs) 
    start_date = first(CPIDataBase.index_dates(cs))
    num_varbase_periods = map(periods, cs.base)
    # Compute new start and end dates
    new_end_dates = map(nperiods -> start_date + Month(nperiods-1), cumsum(num_varbase_periods))
    new_start_dates = map((date, nperiods) -> date - Month(nperiods-1), new_end_dates, num_varbase_periods)
    # Create a new CountryStructure
    cstype = getunionalltype(cs)
    varbases = map(cs.base, new_start_dates, new_end_dates) do base, start_date, end_date
        return VarCPIBase( base.v, base.w, start_date:Month(1):end_date, base.baseindex)
    end
    return cstype(varbases)
end