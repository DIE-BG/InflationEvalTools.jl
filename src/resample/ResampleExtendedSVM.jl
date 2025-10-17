## Resampling of CPIDataBase objects
# A ResampleFunction is defined to implement the interface to VarCPIBase and CountryStructure

# Definition of the resampling function by month of occurrence
"""
    ResampleExtendedSVM <: ResampleFunction
♦
Defines a resampling function to resample the time series by the
same months of occurrence. The sampling is performed independently for each
time series in the columns of a matrix. 

    Exaplme

    resamplefn = ResampleExtendedSVM([150, 180, 50])
    resamplefn(GTDATA23)

    The GTDATA23 contains trhee VarCPIBase, the same length as the vector 
    used as a parameter in ResampleExtendedSVM()
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