using CPIDataGT
CPIDataGT.load_tree_data()

# Finds the indices of codes in the FullCPIBase
findcode(code::AbstractString, base::FullCPIBase) = findfirst(==(code), base.codes)

# Values observed in the CPI base 2010
function distrv10(code_b10::AbstractString, m::Int)
    # Get the tree of code in the CPI base 2010
    branch10 = getb10tree(code_b10)
    # Monthly price changes in CPI base 2010
    v_group = varinterm(compute_index(branch10))
    v_10 = v_group[m:12:end]
    v_10
end

# Values observed in the CPI base 2023
function distrv23(code_b23::AbstractString, m::Int)
    i_23 = findcode(code_b23, FGT23)
    v_23 = distrv23(i_23, m)
    v_23
end
function distrv23(i_23::Int, m::Int)
    # Define the index for the CPI base 2024
    if i_23 >= 17 
        i_24 = i_23 - 1
    else
        i_24 = i_23 
    end
    # Get available observations
    m_available = periods(GT24)
    if (m <= m_available)
        v_23 = [FGT23.v[m:12:end, i_23]..., FGT24.v[m:12:end, i_24]...]
    else
        v_23 = FGT23.v[m:12:end, i_23]
    end
    v_23
end

# Returns a tree with the specified code
function getb10tree(code_b10::AbstractString)
    branch10 = CPITREE10[code_b10]
    if branch10 === nothing
        error("CPI base 2010 code provided not found")
    end
    branch10
end

function get_obs_month(code_b10::AbstractString, code_b23::AbstractString, m::Int)
    # Monthly price changes in CPI base 2010
    v_10 = distrv10(code_b10, m)
    # Monthly price changes in CPI base 2023
    v_23 = distrv23(code_b23, m)
    v_10, v_23
end

function get_names(code_b10::AbstractString, code_b23::AbstractString)
    # Get the tree of code in the CPI base 2010
    branch10 = getb10tree(code_b10)
    name_10 = branch10.tree.name
    i_23 = findcode(code_b23, FGT23)
    name_23 = FGT23.names[i_23]
    name_10, name_23
end

# Helper to get the data and the names of the distributions
function get_names_obs_month(code_b10::AbstractString, code_b23::AbstractString, m::Int)
    # Get observations 
    v_10, v_23 = get_obs_month(code_b10, code_b23, m)
    # Get the names 
    name_10, name_23 = get_names(code_b10, code_b23)

    @info "Prior assignment check" month=m name_10 name_23
    v_10, v_23, name_10, name_23
end

# Mean of the empirical distribution of the CPI base 2010
function get_b10_mean(code_b10, m)
    v_10 = distrv10(code_b10, m)
    mean_v10 = mean(v_10)
    mean_v10
end

# Mean of the empirical distribution of the CPI base 2023
function get_b23_mean(code_b10, code_b23, m, f)
    # Compute the mean of the CPI base 2023 empirical distribution
    vkdistr, wkj, J = get_synth_distr(code_b10, code_b23, m, f)
    mean_v23 = wkj' * vkdistr
    mean_v23
end


# Returns the CPI base 2010 distribution
function get_synth_distr(code_b10::AbstractString, code_b23::AbstractString, m::Int, f::Function)
    v_10, v_23 = get_obs_month(code_b10, code_b23, m)
    J, H = length(v_10), length(v_23)
    vkdistr = [v_10..., v_23...]
    g = f(vkdistr)
    wkj = reweigh(vkdistr, g, H)
    return vkdistr, wkj, J
end