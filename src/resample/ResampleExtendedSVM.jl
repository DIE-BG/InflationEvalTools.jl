## Resampling of CPIDataBase objects
# A ResampleFunction is defined to implement the interface to VarCPIBase and CountryStructure

# Definition of the resampling function by month of occurrence
"""
    ResampleExtendedSVM <: ResampleFunction

Define a resampling function to resample the time series for the same months of occurrence. 
Sampling is performed independently for each time series in the columns of a matrix. 

    Example

    resamplefn = ResampleExtendedSVM([150, 180, 50])
    resamplefn (GTDATA23)

GTDATA23 contains three VarCPIBase, of the same length as the vector used as a parameter 
in ResampleExtendedSVM().
"""
struct ResampleExtendedSVM <: ResampleFunction
    extension_periods::Union{Int,Vector{Int}}
end

# Define how to resample matrices with time series in the columns.
# Uses the internal function `scramblevar`.
function (resamplefn::ResampleExtendedSVM)(cs::CountryStructure, rng=Random.GLOBAL_RNG)
    @Main.infiltrate
    if isa(resamplefn.extension_periods, Vector) &&
       length(resamplefn.extension_periods) != length(cs.base)
        error("The vector of periods must have the same number of VarCPIBase in the CountyStructure")
    end

    if length(resamplefn.extension_periods) == 1
        periods = fill(resamplefn.extension_periods, length(cs.base))
        println(string(resamplefn.extension_periods, "is the number of periods for all VarCPIBase"))
    else
        periods = resamplefn.extension_periods
    end

    # Obtain resampled bases
    ps = (periods...,)
    base_boot = map((b, extension_periods) -> resamplefn(b, extension_periods, rng), cs.base, ps)
    # Return new CountryStructure
    typeof(cs)(base_boot)

end

function (resamplefn::ResampleExtendedSVM)(base::VarCPIBase, extension_periods::Int, rng=Random.GLOBAL_RNG)
    
    v_boot = resamplefn(base.v, extension_periods, rng)
    # Extend the dates
    startdate = base.dates[1]
    dates = startdate:Month(1):Date(startdate + Month(extension_periods - 1))

    return (VarCPIBase(v_boot, base.w, dates, base.baseindex))
end

function (resamplefn::ResampleExtendedSVM)(base::VarCPIBase, rng=Random.GLOBAL_RNG)
    @Main.infiltrate
    v_boot = resamplefn(base.v, resamplefn.extension_periods, rng)
    # Extend the dates
    startdate = base.dates[1]
    dates = startdate:Month(1):Date(startdate + Month(resamplefn.extension_periods - 1))

    return (VarCPIBase(v_boot, base.w, dates, base.baseindex))
end

function (resamplefn::ResampleExtendedSVM)(vmat::AbstractMatrix, extension_periods::Int, rng=Random.GLOBAL_RNG)
    # Number of periods and basic expenditures of the input matrix
    periods_vmat, n = size(vmat)

    # Matrix of resampled values
    v_sc = Matrix{eltype(vmat)}(undef, extension_periods, n)

    # For each month and each basic expenditure, randomly take from the same
    # months of vmat and fill v_sc (v_scrambled)
    for i in 1:min(extension_periods, 12), g in 1:n
        Random.rand!(rng, view(v_sc, i:12:extension_periods, g), view(vmat, i:12:periods_vmat, g))
    end
    v_sc
end

# Define the name and tag of the resampling method 
method_name(resamplefn::ResampleExtendedSVM) = "Extended IID bootstrap by months of occurrence"
method_tag(resamplefn::ResampleExtendedSVM) = "ESVM"
# Define which function to use to obtain the population CPI datasets
get_param_function(::ResampleExtendedSVM) = param_scramblevar_fn
