## ----------------------------------------------------------------------------
#  Mixture resampling function
#
#  A ResampleFunction that combines multiple resampling functions to apply different
#  resampling methods to different VarCPIBase objects within a CountryStructure.
#  Each VarCPIBase object in the CountryStructure is resampled using its
#  corresponding resampling function from the provided collection.
## ----------------------------------------------------------------------------

import Random

"""
    ResampleMixture(resampling_functions::Vector{<:ResampleFunction})

Creates a resampling function that applies different resampling methods to each
VarCPIBase object in a CountryStructure. The number of resampling functions provided
must match the number of VarCPIBase objects in the CountryStructure.

# Arguments
- `resampling_functions`: A vector of ResampleFunction objects, where each element
  corresponds to the resampling function to be applied to the VarCPIBase object at
  the same index in a CountryStructure.

# Example
```julia
# Create different resampling functions for each base
resample_identity = ResampleIdentity()
resample_synthetic = ResampleSynthetic(base, matching_array)

# Create a mixed resampler that applies each function to its corresponding base
mixed_resampler = ResampleMixture([resample_identity, resample_synthetic])

# Apply the mixed resampling to a CountryStructure with two bases
resampled_cs = mixed_resampler(cs)

# Get the population parameters using each sampler's param function
population_fn = get_param_function(mixed_resampler)
population_cs = population_fn(cs)
```
"""
struct ResampleMixture{T <: Vector{<:ResampleFunction}} <: ResampleFunction
    resampling_functions::T
end

method_name(::ResampleMixture) = "Mixture of Resampling Methods"
method_tag(::ResampleMixture) = "MIX"

# Validate that the number of resampling functions matches the number of bases
function _validate_mixture_params(cs::CountryStructure, resampling_functions)
    n_bases = length(cs.base)
    n_functions = length(resampling_functions)

    if n_bases != n_functions
        error("Number of resampling functions ($n_functions) must match number of bases ($n_bases)")
    end

    return true
end

# Implement resampling for CountryStructure by applying each function to its base
function (resamplefn::ResampleMixture)(cs::CountryStructure, rng::AbstractRNG = Random.GLOBAL_RNG)
    _validate_mixture_params(cs, resamplefn.resampling_functions)

    # Apply each resampling function to its corresponding base
    bootstrap_varbases = map(zip(cs.base, resamplefn.resampling_functions)) do (base, fn)
        fn(base, rng)
    end

    # Return new CountryStructure with resampled bases
    cstype = getunionalltype(cs)
    bootstrap_cs = cstype(bootstrap_varbases...)
    bootstrap_cs = _fix_countrystructure_dates(bootstrap_cs)
    return bootstrap_cs
end

# Implement resampling for VarCPIBase using the first resampling function
# This allows the ResampleMixture to be used as a regular ResampleFunction
function (resamplefn::ResampleMixture)(base::VarCPIBase, rng::AbstractRNG = Random.GLOBAL_RNG)
    # Use the first resampling function when applied to a single VarCPIBase
    first_fn = first(resamplefn.resampling_functions)
    return first_fn(base, rng)
end

"""
    get_param_function(resamplefn::ResampleMixture)

Return a function that, when called with a CountryStructure, applies each
resampling function's parameter function to its corresponding base. The returned
function creates a new CountryStructure with the parameter data from each base.
"""
function get_param_function(resamplefn::ResampleMixture)
    # Get the parameter function for each resampler
    param_functions = map(get_param_function, resamplefn.resampling_functions)

    # Build a function that applies each population data function to its
    # corresponding base
    function param_mixture_fn(cs::CountryStructure)
        _validate_mixture_params(cs, param_functions)

        # Apply each parameter function to its corresponding base
        population_varbases = map(zip(cs.base, param_functions)) do (base, fn)
            fn(base)
        end

        # Return new CountryStructure with population bases
        cstype = getunionalltype(cs)
        population_cs = cstype(population_varbases...)
        population_cs = _fix_countrystructure_dates(population_cs)
        return population_cs
    end

    return param_mixture_fn
end
