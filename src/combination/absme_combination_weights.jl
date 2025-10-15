"""
    absme_combination_weights(tray_infl::AbstractArray{F, 3}, tray_infl_param; 
        restrict_all::Bool = true, 
        show_status::Bool = false) where F -> Vector{F}

Obtains non-negative weights, whose sum is equal to 1, for the linear
combination problem that minimizes the mean absolute error value. These
weights can be interpreted as shares in the linear combination.

Optional parameters: 
- `show_status::Bool = false`: show the status of the optimization process with
  Ipopt and JuMP. 
- `restrict_all::Bool = true`: indicates whether to apply the sum-to-one
  constraint to all entries of the weights vector. If `false`, the constraint is
  applied from the second entry. This is so that if the first weight corresponds
  to a constant term, it is not constrained. 
"""
function absme_combination_weights(
    tray_infl::AbstractArray{F, 3}, tray_infl_param; 
    restrict_all::Bool = true, 
    show_status::Bool = false) where F
  
      # Inputs for the absolute value loss function
      n = size(tray_infl, 2)
      ē = vec(mean(tray_infl .- tray_infl_param, dims=[1,3]))
  
      # If restrict_all == false, the sum-to-one constraint is applied from the
      # second position of the weights vector β
      r = restrict_all ? 1 : 2
  
      # Constrained optimization problem
      model = Model(Ipopt.Optimizer)
      @variable(model, β[1:n] >= 0)
      @constraint(model, sum(β[r:n]) == 1)
  
      # Mean error of the combination as the combination of mean errors
      @variable(model, me)
      @constraint(model, me == dot(ē, β))
  
      # Absolute value constraint
      @variable(model, absme)
      @constraints(model, begin absme >= me; absme >= -me end)
  
      @objective(model, Min, absme)
  
      # Obtain the numerical solution 
      show_status || set_silent(model)
      optimize!(model)
      convert.(F, JuMP.value.(β))
  end