# scramblevar.jl - Functions to resample VarCPIBase objects

import Random: AbstractRNG

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
function scramblevar(vmat::AbstractMatrix, rng::AbstractRNG = Random.GLOBAL_RNG, sample_periods::Int = size(vmat, 1))
    # periods = number of periods to resample
    # Actual available periods in vmat
    actual_periods, n = size(vmat)

    # Matrix of resampled values
    v_sc = similar(vmat, sample_periods, n)

    # For each month and each basic expenditure, randomly take from the same
    # months of vmat and fill v_sc (v_scrambled)
    for i in 1:min(sample_periods, 12), g in 1:n
        Random.rand!(rng, view(v_sc, i:12:sample_periods, g), view(vmat, i:12:actual_periods, g))
    end
    return v_sc
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

# Define which function to use to obtain the population CPI datasets
get_param_function(::ResampleScrambleVarMonths) = param_scramblevar_fn

# Define how to resample matrices with time series in the columns.
# Uses the internal function `scramblevar`.
function (resamplefn::ResampleScrambleVarMonths)(vmat::AbstractMatrix, rng = Random.GLOBAL_RNG)
    return scramblevar(vmat, rng)
end

# Define the name and tag of the resampling method
method_name(resamplefn::ResampleScrambleVarMonths) = "IID bootstrap by months of occurrence"
method_tag(resamplefn::ResampleScrambleVarMonths) = "SVM"


"""
    param_scramblevar_fn(base::VarCPIBase)

Obtains the matrix of population monthly price changes for the B-TIMA
bootstrap resampling methodology by the same calendar months. Returns a
`VarCPIBase` type base with the average month-to-month variations of the same
months of occurrence (also called population monthly price changes).

"""
function param_scramblevar_fn(base::VarCPIBase, sample_periods::Int = periods(base))
    # Get the matrix of average monthly price changes
    month_mat = monthavg(base.v, sample_periods)
    # Form the VarCPIbase with monthly averages 
    return VarCPIBase(month_mat, base.w, base.dates, base.baseindex)
end

function param_scramblevar_fn(cs::CountryStructure)
    pob_base = map(param_scramblevar_fn, cs.base)
    return getunionalltype(cs)(pob_base)
end

# Obtener variaciones intermensuales promedio de los mismos meses de ocurrencia.
# Se remuestrean `sample_periods` observaciones de las series de tiempo en las
# columnas de `vmat`.
function monthavg(vmat::AbstractMatrix, sample_periods::Int = size(vmat, 1))
    # Crear la matriz de promedios
    cols = size(vmat, 2)
    avgmat = similar(vmat, sample_periods, cols)

    # Llenar la matriz de promedios con los promedios de cada mes
    for i in 1:12
        avgmat[i:12:end, :] .= mean(vmat[i:12:end, :], dims = 1)
    end
    return avgmat
end
