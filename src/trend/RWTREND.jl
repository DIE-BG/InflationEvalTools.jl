# Directory where the random walk trajectories are stored.
RWTREND_DIR = joinpath(@__DIR__, "..", "..", "data", "RWTREND")

# Vector with all available trajectories.
avaible_rwtrend = map(readdir(RWTREND_DIR)) do x
    @chain x begin
        match(r"\s(.*)\.jld2", _).captures[1]
        Date(_)
    end
end

# The most recent one is chosen to be added to the stochastic trend.
last_avaible_rwtrend = last(sort(avaible_rwtrend))
delog_rwtrend = load(joinpath(RWTREND_DIR, "RWTREND $(last_avaible_rwtrend).jld2"))["RWTREND"]

"""
    RWTREND
Precalibrated random walk trajectory for 292 periods.
"""
const RWTREND = @. convert(Float32, exp(delog_rwtrend))