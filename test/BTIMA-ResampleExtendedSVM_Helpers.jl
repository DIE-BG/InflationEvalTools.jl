# resample.jl - General structure to apply resampling functions to types
# CountryStructure and VarCPIBase

"""
    abstract type ResampleFunction <: Function end
Abstract type for resampling functions. Each function must at least extend the method
- `resamplefn(vmat::AbstractMatrix, rng)::Matrix` 
to resample a CountryStructure with the functions defined above.

Optionally, if you want to modify the specific behavior of each
resampling function, you must extend the following methods:
- `function (resamplefn::ResampleFunction)(cs::CountryStructure, rng = Random.GLOBAL_RNG)`
- `function (resamplefn::ResampleFunction)(base::VarCPIBase, rng = Random.GLOBAL_RNG)`
"""
abstract type ResampleFunction <: Function end


## Resampling methods for CountryStructure and VarCPIBase

"""
    function (resamplefn::ResampleFunction)(cs::CountryStructure, rng = Random.GLOBAL_RNG)
Defines the general behavior of a resampling function on CountryStructure.
Each of the bases in the `base` field is resampled using the method for `VarCPIBase` objects,
and a new `CountryStructure` is returned.
"""
function (resamplefn::ResampleFunction)(cs::CountryStructure, rng=Random.GLOBAL_RNG)

    # Obtain resampled bases, this requires defining a method to handle
    # objects of type VarCPIBase
    base_boot = map(b -> resamplefn(b, rng), cs.base)

    # Return new CountryStructure with
    typeof(cs)(base_boot)
end


"""
    function (resamplefn::ResampleFunction)(base::VarCPIBase, rng = Random.GLOBAL_RNG)
Defines the general behavior of a resampling function on `VarCPIBase`.
This method requires a specific implementation of the method on the pair (`AbstractMatrix`, rng).
Note that the resampling method could extend the periods of the time series and
adjusts the dates appropriately.
"""
function (resamplefn::ResampleFunction)(base::VarCPIBase, rng=Random.GLOBAL_RNG)

    # Obtain the resampled matrix, requires defining the method to handle
    # matrices
    v_boot = resamplefn(base.v, rng)

    # Set up a new VarCPIBase. Weight vector and base indices
    # unchanged. Dates remain unchanged if the resampling function
    # does not extend the periods in the month-to-month variation matrix
    # `base.v`
    periods = size(v_boot, 1)
    if periods == size(base.v, 1)
        dates = base.dates
    else
        startdate = base.dates[1]
        dates = startdate:Month(1):(startdate+Month(periods - 1))
    end

    VarCPIBase(v_boot, base.w, dates, base.baseindex)
end



"""
    get_param_function(::ResampleFunction)

Each resampling function must implement a function to
obtain a `CountryStructure` with `VarCPIBase` objects that contain the
average (or parametric) month-to-month variations that allow constructing the
parametric inflation trajectory according to the resampling method given
in `ResampleFunction`.

This function returns the function to obtain parametric data.

## Example 
```julia
# Obtain the resampling function
resamplefn = ResampleSBB(36)
...

# Obtain its function to get the parametric data
paramdatafn = get_param_function(resamplefn)
# Obtain CountryStructure of parametric data
paramdata = paramdatafn(gtdata)
```

See also: [`param_sbb`](@ref)
"""
get_param_function(::ResampleFunction) =
    error("A function must be specified to obtain the parameter of this resampling function")


"""
    method_name(resamplefn::ResampleFunction)
Function to obtain the name of the resampling method.
"""
method_name(::ResampleFunction) = error("The name of the resampling method must be redefined")

"""
    method_tag(resamplefn::ResampleFunction)
Function to obtain a tag for the resampling method.
"""
method_tag(resamplefn::ResampleFunction) = string(nameof(resamplefn))


############## ResampleExtendedSVM type ##############
## Resampling of CPIDataBase objects
# A ResampleFunction is defined to implement the interface to VarCPIBase and CountryStructure

# Definition of the resampling function by month of occurrence
"""
    ResampleExtendedSVM <: ResampleFunction

Defines a resampling function to resample the time series by the
same months of occurrence and extended it by a custum user parameter. 
The sampling is performed independently for each time series in the columns of a matrix.

    Exaplme

    resamplefn = ResampleExtendedSVM([150, 180, 50])
    
"""
struct ResampleExtendedSVM <: ResampleFunction 
    extended::Vector{Int}
end

# Define how to resample matrices with time series in the columns.
# Uses the internal function `scramblevar`.
function (resamplefn::ResampleExtendedSVM)(cs::CountryStructure, rng = Random.GLOBAL_RNG)

    # Obtain resampled bases
    ps = (resamplefn.extended...,)
    base_boot = map((b, extended) -> resamplefn(b, extended, rng), cs.base, ps)
    # Return new CountryStructure
    typeof(cs)(base_boot)
end

function (resamplefn::ResampleExtendedSVM)(base::VarCPIBase, extended, rng = Random.GLOBAL_RNG)
    v_boot = resamplefn(base.v, extended, rng)
    # Extend the dates
    startdate = base.dates[1]
    dates = startdate:Month(1):Date(startdate + Month(extended - 1))
    VarCPIBase(v_boot, base.w, dates, base.baseindex)
end

function (resamplefn::ResampleExtendedSVM)(vmat::AbstractMatrix, extended, rng = Random.GLOBAL_RNG) 
    # Number of periods and basic expenditures of the input matrix
    periods_vmat, n = size(vmat)
    
    # Matrix of resampled values
    v_sc = Matrix{eltype(vmat)}(undef, extended, n)

    # For each month and each basic expenditure, randomly take from the same
    # months of vmat and fill v_sc (v_scrambled)
    for i in 1:min(extended, 12), g in 1:n 
        Random.rand!(rng, view(v_sc, i:12:extended, g), view(vmat, i:12:periods_vmat, g))        
    end    
    v_sc
end 

# Define the name and tag of the resampling method 
method_name(resamplefn::ResampleExtendedSVM) = "Extended IID bootstrap by months of occurrence"
method_tag(resamplefn::ResampleExtendedSVM) = "ESVM"
# Define which function to use to obtain the population CPI datasets
get_param_function(::ResampleExtendedSVM) = param_scramblevar_fn

"""
    param_extendedscramblevar_fn(base::VarCPIBase)

Obtains the matrix of population monthly price changes for the B-TIMA
bootstrap resampling methodology by the same calendar months. Returns a
`VarCPIBase` type base with the average month-to-month variations of the same
months of occurrence (also called population monthly price changes).

"""
function param_extendedscramblevar_fn(base::VarCPIBase)

    # Obtener matriz de promedios mensuales
    month_mat = monthavg(base.v)

    # Conformar base de variaciones intermensuales promedio
    VarCPIBase(month_mat, base.w, base.dates, base.baseindex)
end

function param_extendedscramblevar_fn(cs::CountryStructure)
    pob_base = map(param_extendedscramblevar_fn, cs.base)
    getunionalltype(cs)(pob_base)
end

# Obtener variaciones intermensuales promedio de los mismos meses de ocurrencia.
# Se remuestrean `numobsresample` observaciones de las series de tiempo en las
# columnas de `vmat`. 
function monthavg(vmat, numobsresample = size(vmat, 1))
    # Crear la matriz de promedios 
    cols = size(vmat, 2)
    avgmat = Matrix{eltype(vmat)}(undef, numobsresample, cols)
    
    # Llenar la matriz de promedios con los promedios de cada mes 
    for i in 1:12
        avgmat[i:12:end, :] .= mean(vmat[i:12:end, :], dims=1)
    end
    return avgmat
end
