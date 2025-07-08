# ResampleGSBBMod.jl - Functions to resample
# VarCPIBase objects with the Generalized Seasonal Block Bootstrap methodology.

# NOTE: a variant with block length = 25 and with 300 output observations is used.
# The more general method described in the paper does not consider the
# extension of the time series.


# Definition of the GSBB resampling function
Base.@kwdef struct ResampleGSBBMod <: ResampleFunction
    blocklength::Int = 25
end

# Define which function to use to obtain parametric bases
get_param_function(::ResampleGSBBMod) = param_gsbb_mod

# Define the name and tag of the resampling method
method_name(resamplefn::ResampleGSBBMod) = "Seasonal block bootstrap with block size " * string(resamplefn.blocklength) 
method_tag(resamplefn::ResampleGSBBMod) = string(nameof(resamplefn)) * "-" * string(resamplefn.blocklength)


# Define how to resample matrices with time series in the columns
function (resample_gsbb_fn::ResampleGSBBMod)(vmat::AbstractMatrix, rng = Random.GLOBAL_RNG)
    G = size(vmat, 2)
    boot_vmat = Matrix{eltype(vmat)}(undef, 300, G)

    # Sampling indices for 25-month blocks
    ids = [(12i + j):(12i + j + 24) for i in 0:7, j in 1:12]

    for j in 1:12
        # Sample a range and assign it in the 25-month block
        range_ = rand(rng, view(ids, :, j))
        boot_vmat[(25(j-1) + 1):(25j), :] = vmat[range_, :]
    end

    boot_vmat
end

# Modify how to resample CountryStructure objects to modify the dates in the resampled bases
function (resample_gsbb_fn::ResampleGSBBMod)(cs::CountryStructure, rng = Random.GLOBAL_RNG)
    # Obtain resampled bases
    base_boot = map(b -> resample_gsbb_fn(b, rng), cs.base)
        
    # Modify the dates of the second base
    finalbase = base_boot[2]
    startdate = base_boot[1].dates[end] + Month(1)
    T = periods(finalbase)
    newdates = getdates(startdate, T)
    base10_mod = VarCPIBase(finalbase.v, finalbase.w, newdates, finalbase.baseindex)

    # Return the CountryStructure again
    CPIDataBase.getunionalltype(cs)(base_boot[1], base10_mod)
end