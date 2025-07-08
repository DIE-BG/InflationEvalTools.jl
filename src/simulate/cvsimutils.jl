# Simulation functions for CrossEvalConfig

# Function for generation of trajectories and parameters for the cross-validation procedure of linear combinations of inflation measures
function makesim(data::CountryStructure, config::CrossEvalConfig; kwargs...)

    # Obtain inflation parameter
    param = InflationParameter(config.paramfn, config.resamplefn, config.trendfn)
    # Results dictionary
    cvinputs = Dict{String, Any}()
    cvinputs["config"] = config
    # Extra options for pargentrayinfl
    pargenkwargs = filter(e -> first(e) != :K, kwargs)
    
    # Generate data for each training and validation subperiod
    for (i, evalperiod) in enumerate(config.evalperiods)
          
        traindate = evalperiod.startdate - Month(1)
        cvdate = evalperiod.finaldate
        fmt = dateformat"yy"
        @info "Cross-validation iteration $i" evalperiod traindate cvdate 
        
        # Generate inflation trajectories and parametric trajectory
        for finaldate in (traindate, cvdate)

            # Obtain the keys to save the results. The format is the
            # prefix "infl_" or "param_" and the last two years of the
            # final date of each training or validation subperiod
            tray_key = "infl_" * Dates.format(finaldate, fmt)
            param_key = "param_" * Dates.format(finaldate, fmt)
            dates_key = "dates_" * Dates.format(finaldate, fmt)
            sliced_data = data[finaldate]

            # Generate inflation trajectories
            if !(tray_key in keys(cvinputs))
                @info "Generating inflation trajectories" finaldate
                cvinputs[tray_key] = pargentrayinfl(
                    config.inflfn, 
                    config.resamplefn, 
                    config.trendfn, 
                    sliced_data; 
                    K = config.nsim, pargenkwargs...)
                cvinputs[dates_key] = infl_dates(sliced_data)
            end

            # Generate parametric trajectory
            if !(param_key in keys(cvinputs)) 
                @info "Generating parametric trajectory" finaldate
                cvinputs[param_key] = param(sliced_data)
            end

        end

    end

    cvinputs
end
