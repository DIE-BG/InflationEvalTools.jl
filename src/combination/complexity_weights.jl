# Complexity-based penalty for the inflation measures

"""
    share_combination_complexity_weights_rmse(tray_infl::AbstractArray{F,3}, tray_infl_param::AbstractArray{F,3}, complexity_factors::AbstractVector{F}, λ::AbstractFloat = 0.0) -> Vector{F}

Compute a vector of non-negative weights `β` that add up to 1 by minimizing a nonlinear loss function based on the average root mean squared error (RMSE).

This function differs from [`share_combination_weights_rmse`](@ref) in that it adds a complexity-based penalty to the loss function. 
The penalty is defined as `100λ * sum(complexity_factors .* β)`, where `complexity_factors` is a vector of complexity relative weights for each inflation trajectory. 
The penalty encourages the optimization to assign higher weights to trajectories with lower complexity, thus promoting more robust, simpler models.

Arguments:
- `tray_infl::AbstractArray{F,3}`: Simulated inflation trajectories.
- `tray_infl_param::AbstractArray{F,3}`: Parametric inflation trajectories.
- `complexity_factors::AbstractVector{F}`: A vector of complexity relative weights for each inflation trajectory.
- `λ::AbstractFloat = 0.0`: Regularization parameter to penalize small weights.

See also: [`share_combination_weights_rmse`](@ref)
"""
function share_combination_complexity_weights_rmse(
        traj_infl::AbstractArray{F, 3},
        traj_infl_param::AbstractArray{F, 3},
        complexity_factors::AbstractVector{F},
        λ::AbstractFloat = 0.0,
    ) where {F}

    T_traj, N_traj, K_traj = size(traj_infl)
    T_param, N_param, N_batches = size(traj_infl_param)

    @assert T_traj == T_param "The trajectories and the parameter should have the same number of periods."
    @assert K_traj % N_batches == 0 "The number of trajectories should be the number of simulations times the numbers of batches."
    @assert length(complexity_factors) == N_traj "The number of complexity factors should be equal to the number of trajectories."

    # number of simulations per batch
    N_sim = convert(Int, K_traj / N_batches)

    function rmse_loss(β)
        # The first level of the loop represents the batch in the trend simulation
        rmse_b = Vector(undef, N_batches)
        for b in 1:N_batches

            # taking only the parameter and simulated trajectories for batch b
            param_b = @view traj_infl_param[:, :, b]
            traj_b = @view traj_infl[:, :, (1 + N_sim * (b - 1)):(N_sim * b)]

            # The second level of the loop represents the simulations per batch
            rmse_k = Vector(undef, N_sim)
            for k in 1:N_sim
                # The third level is the accumulation of the squared error
                ∑e² = F(0)
                for t in 1:T_traj
                    ∑e² += (traj_b[t, :, k]' * β - param_b[t])^2
                end
                # computes N_sim RMSE for batch b
                rmse_k[k] = sqrt((1 / T_traj) * ∑e²)
            end
            # average the k RMSEs for batch b
            rmse_b[b] = mean(rmse_k)
        end

        # average the RMSEs across batches
        return mean(rmse_b)
    end

    # Definition of the penalty function. it takes a mean to ensure a scalar
    # Here we use standard Ridge regularization with different factors for each
    # combination coefficient
    penalty(β) = sum(complexity_factors .* (β .^ 2))

    # having the loss and the penalty, define a penalized loss function for
    # the optimizer
    # Importantly for the calibration: the relative weight between rmse_loss and penalty is determined by λ
    penalized_loss(β) = rmse_loss(β) + (λ * penalty(β))

    # restricted optimization problem
    model = Model(Ipopt.Optimizer)
    @variable(model, β[1:N_traj] >= 0)
    @constraint(model, sum(β[1:N_traj]) == 1)
    @objective(model, Min, penalized_loss(β))
    optimize!(model)

    # extracting the optimal weights
    β_opt = convert.(F, JuMP.value.(β))

    return Dict(
        :value => β_opt,
        :rmse => rmse_loss(β_opt),
        :penalty => penalty(β_opt),
        :penalized_rmse => penalized_loss(β_opt),
    )
end
