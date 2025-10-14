# Versión experimental con configuración de semilla por iteración
function pargentrajinfl_seed(inflfn::InflationFunction, csdata::CountryStructure; 
    trj = 100, rndseed = 161803, showprogress = true)

    # Matriz de trayectorias de salida
    T = sum(size(gtdata[i].v, 1) for i in 1:length(gtdata.base)) - 11
    traj_infl = SharedArray{Float32}(T, trj)

    # Generar las trayectorias
    @sync @showprogress @distributed for i in 1:trj 
        
        # Replicación simulación por simulación
        Random.seed!(rndseed + i)
        
        # Muestra de bootstrap de los datos 
        bootsample = deepcopy(csdata)
        scramblevar!(bootsample)

        # Computar la medida de inflación 
        traj_infl[:, i] = inflfn(bootsample)
    end

    # Retornar las trayectorias
    sdata(traj_infl)
end


# Versión experimental con opción de control de progreso
function pargentrajinfl_prog(inflfn::InflationFunction, csdata::CountryStructure; 
    trj = 100, rndseed = 161803, showprogress = true)

    # Configurar la semilla en workers
    remote_seed(rndseed)

    # Matriz de trayectorias de salida
    T = sum(size(gtdata[i].v, 1) for i in 1:length(gtdata.base)) - 11
    traj_infl = SharedArray{Float32}(T, trj)

    # Control de progreso
    p = Progress(trj; enabled = true)
    channel = RemoteChannel(() -> Channel{Bool}(trj), 1)

    @sync begin 
        # this task prints the progress bar
        @async while take!(channel)
            next!(p)
        end
    
        # Esta tarea genera las trayectorias
        @async begin 
            @distributed for i in 1:trj 
                # Muestra de bootstrap de los datos 
                bootsample = deepcopy(csdata)
                scramblevar!(bootsample)

                # Computar la medida de inflación 
                traj_infl[:, i] = inflfn(bootsample)
                
                # ProgressMeter
                put!(channel, true)
            end
            put!(channel, false) # esto avisa a la tarea que se terminó
        end
    end
    # Retornar las trayectorias
    sdata(traj_infl)
end


# export pargentrayinfl_pmap
# Versión con pmap es más lenta
function pargentrajinfl_pmap(inflfn::F, csdata::CS; 
    trj = 100, rndseed = 161803, showprogress = true) where {F <: InflationFunction, CS <: CountryStructure}

    p = Progress(trj, barglyphs=BarGlyphs("[=> ]"), enabled = showprogress)
    
    traj_infl = progress_pmap(1:trj, progress=p) do k
        # Configurar la semilla en el proceso
        Random.seed!(LOCAL_RNG, rndseed + k)

        # Muestra de bootstrap de los datos 
        bootsample = deepcopy(csdata)
        scramblevar!(bootsample, LOCAL_RNG)

        # Computar la medida de inflación 
        inflfn(bootsample)
    end

    # Retornar las trayectorias
    # cat(tray_infl...; dims=3)
    traj_infl
end