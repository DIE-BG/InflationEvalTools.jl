# param.jl - Development of functions to obtain parametric inflation trajectory

"""
    param_gsbb_mod(base::VarCPIBase)
Obtains the matrix of population monthly price changes for the
Generalized Seasonal Block Bootstrap resampling methodology, modified to
extend the observations to 300 periods. Returns a VarCPIBase type base with
the population monthly price changes. Currently works only if `base`
has 120 observations.
"""
function param_gsbb_mod(base::VarCPIBase)

    G = size(base.v, 2)
    vpob = zeros(eltype(base), 300, G)

    # Sampling indices for 25-month blocks
    ids = [(12i + j):(12i + j + 24) for i in 0:7, j in 1:12]

    # Obtain averages
    for m in 1:12
        # Obtain matrices at the corresponding indices
        month_mat = map(range_ -> base.v[range_, :], ids[:, m])

        # Obtain average of month matrices and assign it to the matrix of
        # month-to-month variations for construction of the parametric
        # inflation trajectory
        vpob[(25(m-1) + 1):(25m), :] = mean(month_mat)
    end

    # Set up dates
    dates = getdates(first(base.dates), vpob)
    VarCPIBase(vpob, base.w, dates, base.baseindex)
end

"""
    param_gsbb_mod(cs::CountryStructure)
Obtains a parametric `CountryStructure`.
"""
function param_gsbb_mod(cs::CountryStructure)
    # Obtain population bases
    pob_base = map(param_gsbb_mod, cs.base)
    
    # Modify the dates of the second base
    finalbase = pob_base[2]
    startdate = pob_base[1].dates[end] + Month(1)
    T = periods(finalbase)
    newdates = getdates(startdate, T)
    base10_mod = VarCPIBase(finalbase.v, finalbase.w, newdates, finalbase.baseindex)

    # Set up new CountryStructure with population bases
    getunionalltype(cs)(pob_base[1], base10_mod)
end



"""
    param_sbb(base::VarCPIBase)
Obtains the matrix of population monthly price changes for the
Stationary Block Bootstrap resampling methodology. Returns a `VarCPIBase`
type base with the average month-to-month variations of the same months of
occurrence (also called population monthly price changes).

This definition also applies to other methodologies that use as parametric
month-to-month variations the averages in the same months of occurrence.
"""
function param_sbb(base::VarCPIBase)

    # Obtain matrix of monthly averages
    month_mat = monthavg(base.v)

    # Set up base of average month-to-month variations
    VarCPIBase(month_mat, base.w, base.dates, base.baseindex)
end


"""
    param_sbb(cs::CountryStructure)
Obtains a parametric `CountryStructure`.
See also [`param_sbb`](@ref param_sbb).
"""
function param_sbb(cs::CountryStructure)
    pob_base = map(param_sbb, cs.base)
    getunionalltype(cs)(pob_base)
end


