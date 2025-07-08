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
function (resamplefn::ResampleFunction)(cs::CountryStructure, rng = Random.GLOBAL_RNG)
    
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
function (resamplefn::ResampleFunction)(base::VarCPIBase, rng = Random.GLOBAL_RNG)

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
        dates = startdate:Month(1):(startdate + Month(periods - 1))
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

