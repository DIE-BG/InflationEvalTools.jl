# scramblevar.jl - Functions to resample VarCPIBase objects

# Esta es la mejor versión, requiere crear copias de los vectores de los mismos
# meses, para cada gasto básico. Se presenta una versión más eficiente abajo
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

# Primera versión con remuestreo por columnas 
# function scramblevar(vmat::AbstractMatrix, rng = Random.GLOBAL_RNG) 
#     periods, n = size(vmat)
#     # Matriz de valores remuestreados 
#     v_sc = similar(vmat) 
#     for i in 1:min(periods, 12)
#         v_month = vmat[i:12:periods, :]
#         periods_month = size(v_month, 1)
#         for g in 1:n 
#             v_month[:, g] = rand(rng, v_month[:, g], periods_month)
#         end       
#         # Asignar valores de los mismos meses
#         v_sc[i:12:periods, :] = v_month
#     end
#     v_sc
# end

# Versión optimizada para memoria 
# 420.100 μs (2 allocations: 204.45 KiB)
function scramblevar(vmat::AbstractMatrix, rng = Random.GLOBAL_RNG) 
    periods, n = size(vmat)

    # Matriz de valores remuestreados 
    v_sc = similar(vmat) 

    # Para cada mes y cada gasto básico, tomar aleatoriamente de los mismos
    # meses de vmat y llenar v_sc (v_scrambled)
    for i in 1:min(periods, 12), g in 1:n 
        Random.rand!(rng, view(v_sc, i:12:periods, g), view(vmat, i:12:periods, g))        
    end    
    v_sc
end


## Remuestreo de objetos de CPIDataBase
# Se define una ResampleFunction para implementar interfaz a VarCPIBase y CountryStructure

# Definición de la función de remuestreo por ocurrencia de meses
"""
    ResampleScrambleVarMonths <: ResampleFunction

Define una función de remuestreo para remuestrear las series de tiempo por los
mismos meses de ocurrencia. El muestreo se realiza de manera independiente para 
serie de tiempo en las columnas de una matriz. 
"""
struct ResampleScrambleVarMonths <: ResampleFunction end

# Definir cuál es la función para obtener bases paramétricas 
get_param_function(::ResampleScrambleVarMonths) = param_scramblevar_fn

# Define cómo remuestrear matrices con las series de tiempo en las columnas.
# Utiliza la función interna `scramblevar`.
function (resamplefn::ResampleScrambleVarMonths)(vmat::AbstractMatrix, rng = Random.GLOBAL_RNG) 
    scramblevar(vmat, rng)
end 

# Definir el nombre y la etiqueta del método de remuestreo 
method_name(resamplefn::ResampleScrambleVarMonths) = "Bootstrap IID por meses de ocurrencia"
method_tag(resamplefn::ResampleScrambleVarMonths) = "SVM"


#     param_scramblevar_fn(base::VarCPIBase)
#
# Obtiene la matriz de variaciones intermensuales paramétricas para la
# metodología de remuestreo de por meses de ocurrencia. Devuelve una base de
# tipo `VarCPIBase` con las variaciones intermensuales promedio de los mismos meses
# de ocurrencia (también llamadas variaciones intermensuales paramétricas). 
#
# Esta definición también aplica a otras metodologías que utilicen como variaciones 
# intermensuales paramétricas los promedios en los mismos meses de ocurrencia. 
function param_scramblevar_fn(base::VarCPIBase)

    # Obtener matriz de promedios mensuales
    month_mat = monthavg(base.v)

    # Conformar base de variaciones intermensuales promedio
    VarCPIBase(month_mat, base.w, base.dates, base.baseindex)
end

function param_scramblevar_fn(cs::CountryStructure)
    pob_base = map(param_scramblevar_fn, cs.base)
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
