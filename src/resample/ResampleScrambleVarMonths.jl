# scramblevar.jl - Functions to resample VarCPIBase objects

# This is the best version, it requires creating copies of the vectors of the same
# months, for each basic expenditure. A more efficient version is presented below
# 475.600 μs (2618 allocations: 613.20 KiB)

# function scramblevar!(vmat::AbstractMatrix, rng = Random.GLOBAL_RNG) 
#     for i in 1:12
#         # fill every column with random values from the same periods (t and t+12)
#         for j in 1:size(vmat, 2)
#             rand!(rng, (@view vmat[i:12:end, j]), vmat[i:12:end, j])
#         end
#     end
# end


# function scramblevar(vmat::AbstractMatrix, rng = Random.GLOBAL_RNG) 
#     scrambled_mat = copy(vmat)
#     scramblevar!(scrambled_mat, rng)
#     scrambled_mat
# end

# First version with column resampling 
# function scramblevar(vmat::AbstractMatrix, rng = Random.GLOBAL_RNG) 
#     periods, n = size(vmat)
#     # Matrix of resampled values 
#     v_sc = similar(vmat) 
#     for i in 1:min(periods, 12)
#         v_month = vmat[i:12:periods, :]
#         periods_month = size(v_month, 1)
#         for g in 1:n 
#             v_month[:, g] = rand(rng, v_month[:, g], periods_month)
#         end       
#         # Assign values of the same months
#         v_sc[i:12:periods, :] = v_month
#     end
#     v_sc
# end

# Memory-optimized version 
# 420.100 μs (2 allocations: 204.45 KiB)
function scramblevar(vmat::AbstractMatrix, rng = Random.GLOBAL_RNG) 
    periods, n = size(vmat)

    # Matrix of resampled values 
    v_sc = similar(vmat) 

    # For each month and each basic expenditure, randomly take from the same
    # months of vmat and fill v_sc (v_scrambled)
    for i in 1:min(periods, 12), g in 1:n 
        Random.rand!(rng, view(v_sc, i:12:periods, g), view(vmat, i:12:periods, g))        
    end    
    v_sc
end


## Resampling of CPIDataBase objects
# A ResampleFunction is defined to implement the interface to VarCPIBase and CountryStructure

# Definition of the resampling function by month of occurrence
"""
    ResampleScrambleVarMonths <: ResampleFunction

Defines a resampling function to resample the time series by the
same months of occurrence. The sampling is performed independently for each
time series in the columns of a matrix.
"""
struct ResampleScrambleVarMonths <: ResampleFunction end

# Define which function to use to obtain parametric bases 
get_param_function(::ResampleScrambleVarMonths) = param_sbb

# Define how to resample matrices with time series in the columns.
# Uses the internal function `scramblevar`.
function (resamplefn::ResampleScrambleVarMonths)(vmat::AbstractMatrix, rng = Random.GLOBAL_RNG) 
    scramblevar(vmat, rng)
end 

# Define the name and tag of the resampling method 
method_name(resamplefn::ResampleScrambleVarMonths) = "IID bootstrap by months of occurrence"
method_tag(resamplefn::ResampleScrambleVarMonths) = "SVM"