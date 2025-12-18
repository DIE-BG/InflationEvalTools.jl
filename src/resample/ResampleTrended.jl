# ResampleTrended.jl - Functions to compute the methodology of
# resampling by months of occurrence, with probabilistic weights to recreate the
# trend of the data. The parameter p is individual for each base of a
# CountryStructure

"""
    ResampleTrended{T<:AbstractFloat} <: ResampleFunction

Resampling function that uses IID bootstrap weighted by months of occurrence
with individual parameters for each CPI base, to recreate the trend of the data.

# Fields
- `p::Vector{T}`: Individual probability parameters for each CPI base, controlling
  the trend weighting in the resampling process.
"""
struct ResampleTrended{T<:AbstractFloat} <: ResampleFunction
    p::Vector{T}
end

method_name(fn::ResampleTrended) = "IID bootstrap weighted by months of occurrence, individual bases"
method_tag(fn::ResampleTrended) = "RSTI"
get_param_function(fn::ResampleTrended) = cs -> param_rst(cs, fn.p)

# Overload this method to operate the resample function by base
function (resamplefn::ResampleTrended)(cs::CountryStructure, rng = Random.GLOBAL_RNG)
    # Obtain resampled bases
    ps = (resamplefn.p...,)
    base_boot = map((b, p) -> resamplefn(b, p, rng), cs.base, ps)
    # Return new CountryStructure
    typeof(cs)(base_boot)
end

function (resamplefn::ResampleTrended)(base::VarCPIBase, p, rng = Random.GLOBAL_RNG)
    v_boot = resamplefn(base.v, p, rng)
    VarCPIBase(v_boot, base.w, base.dates, base.baseindex)
end

# Definition to operate over arbitrary matrix
function (resamplefn::ResampleTrended)(vmat::AbstractMatrix, p, rng = Random.GLOBAL_RNG)
    periods, ngoods = size(vmat)
    indexes = Vector{Int}(undef, periods)
    
    # Create and return the resampled series
    resampled_vmat = similar(vmat)
    # Procedure of weighted resampling is applied for every good or service in
    # the vmat matrix
    for j in 1:ngoods
        trended_inds!(indexes, p, rng)
        resampled_vmat[:, j] .= vmat[indexes, j]
    end
    resampled_vmat
end

##  ----------------------------------------------------------------------------
#   Auxiliary functions to define the parametric dataset
#   ----------------------------------------------------------------------------

function param_rst(cs::CountryStructure, p::Vector{<:AbstractFloat})
    # Resample each CPI dataset in the CountryStructure
    ps = (p...,)
    pob_base = map((b,p) -> param_rst(b, p), cs.base, ps)
    getunionalltype(cs)(pob_base)
end