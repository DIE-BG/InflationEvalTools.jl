## ----------------------------------------------------------------------------
#  Identity resampling function
#
#  A trivial ResampleFunction that returns the input datasets unchanged.
#  Useful as a baseline or when no resampling is desired. The identity
#  resampler implements the same interface as other resamplers: it is callable
#  on `CountryStructure` and `VarCPIBase` objects, provides `method_name` and
#  `method_tag`, and exposes `get_param_function` which returns a function that
#  returns the population `VarCPIBase` (identity behaviour).
## ----------------------------------------------------------------------------

import Random

"""
    ResampleIdentity()

An identity resampling function type. Calling an instance of `ResampleIdentity`
on a `VarCPIBase` or `CountryStructure` returns the exact same object (no
resampling performed). Use this when you want to disable resampling and work
with the original observed datasets.
"""
struct ResampleIdentity <: ResampleFunction end

method_name(::ResampleIdentity) = "Identity Resampling (no-op)"
method_tag(::ResampleIdentity) = "IDTY"

# Return the input `CountryStructure` unchanged. This keeps the same object
# reference (no copy) to emphasize identity behaviour; callers that require a
# copy should explicitly clone the object before calling.
function (resamplefn::ResampleIdentity)(cs::CountryStructure, rng::AbstractRNG=Random.GLOBAL_RNG)
    return cs
end

# Return the input `VarCPIBase` unchanged. The identity resampler does not perform
# any bootstrapping: the observed matrix of monthly price changes is returned as
# the resampled matrix.
function (resamplefn::ResampleIdentity)(base::VarCPIBase, rng::AbstractRNG=Random.GLOBAL_RNG)
    return base
end

"""
    get_param_function(resamplefn::ResampleIdentity)

Return a function that, when called with a `VarCPIBase`, returns the same
`VarCPIBase` object (the actual data). This matches the `get_param_function`
contract used by other resamplers, with the exception of returning actual data.
"""
get_param_function(::ResampleIdentity) = identity
