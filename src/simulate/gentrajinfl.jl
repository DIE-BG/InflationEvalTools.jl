# Función de generación de trayectorias de inflación sin computación paralela

"""
    gentrayinfl(inflfn::F, resamplefn::R, trendfn::T, csdata::CountryStructure; 
        K = 100, 
        rndseed = DEFAULT_SEED, 
        showprogress = true)

Computa `K` trayectorias de inflación utilizando la función de inflación
`inflfn::``InflationFunction`, la función de remuestreo
`resamplefn::``TrendFunction` y la función de tendencia
`trendfn::``TrendFunction` especificada. Se utilizan los datos en el
`CountryStructure` dado en `csdata`.

A diferencia de la función [`pargentrayinfl`](@ref), esta función no realiza el
cómputo de forma  distribuida. 

Para lograr la reproducibilidad entre diferentes corridas de la función, y de
esta forma, generar trayectorias de inflación con diferentes metodologías
utilizando los mismos remuestreos, se fija la semilla de generación de acuerdo
con el número de iteración en la simulación. Para controlar el inicio de la
generación de trayectorias se utiliza como parámetro de desplazamiento el valor
`rndseed`, cuyo valor por defecto es la semilla [`DEFAULT_SEED`](@ref). 
"""
function gentrajinfl(inflfn::F, resamplefn::R, trendfn::T, 
    csdata::CountryStructure; 
    trj = 100, 
    rndseed = DEFAULT_SEED, 
    showprogress = true) where {F <: InflationFunction, R <: ResampleFunction, T <: TrendFunction}

    # Configurar el generador de números aleatorios
    myrng = MersenneTwister(rndseed)

    # Cubo de trayectorias de salida
    periods = infl_periods(csdata)
    n_measures = num_measures(inflfn)
    tray_infl = zeros(Float32, periods, n_measures, trj)

    # Control de progreso
    p = Progress(trj; enabled = showprogress)

    # Generar las trayectorias
    for i in 1:trj 
        # Muestra de bootstrap de los datos 
        bootsample = resamplefn(csdata, myrng)
        # Aplicación de la función de tendencia 
        trended_sample = trendfn(bootsample)

        # Computar la medida de inflación 
        tray_infl[:, :, i] = inflfn(trended_sample)
        
        ProgressMeter.next!(p)
    end

    # Retornar las trayectorias
    traj_infl
end