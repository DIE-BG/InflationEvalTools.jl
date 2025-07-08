# EvalPeriod.jl - Type to represent evaluation periods in the simulation exercise

"""
    abstract type AbstractEvalPeriod    
Abstract type to represent types of evaluation periods.

See also: [`EvalPeriod`](@ref), [`CompletePeriod`](@ref).
"""
abstract type AbstractEvalPeriod end 

"""
    EvalPeriod <: AbstractEvalPeriod
Type to represent an evaluation period given by the dates `startdate` and
`finaldate`. A label must be included in the `tag` field to attach to the
results generated in [`evalsim`](@ref). This period can be provided to a
[`SimConfig`](@ref) configuration to evaluate over a specific date range.

## Example

We create an evaluation period called `b2010` when generating the results.
```
julia> b2010 = EvalPeriod(Date(2011,1), Date(2019,12), "b2010")
b2010:Jan-11-Dec-19
```

See also: [`GT_EVAL_B00`](@ref), [`GT_EVAL_B10`](@ref), [`GT_EVAL_T0010`](@ref)
"""
struct EvalPeriod <: AbstractEvalPeriod
    startdate::Date 
    finaldate::Date 
    tag::String 
end

"""
    CompletePeriod <: AbstractEvalPeriod
Type to represent the complete evaluation period, corresponding to the
inflation periods of the data `CountryStructure`. The default `tag` for
the complete period is empty (`""`), so that the evaluation metrics in the
results generated in [`evalsim`](@ref) do not have a prefix, as it is the
main evaluation period. This period can be provided to a
[`SimConfig`](@ref) configuration to evaluate over the entire range of
simulated inflation dates.

## Example

We create an instance of this type to represent the evaluation over the
complete period of the inflation trajectories generated in the simulations.
```
julia> comp = CompletePeriod()
Complete period
```

See also: [`EvalPeriod`](@ref), [`GT_EVAL_B00`](@ref), [`GT_EVAL_B10`](@ref),
[`GT_EVAL_T0010`](@ref)
"""
struct CompletePeriod <: AbstractEvalPeriod
end

"""
    eval_periods(cs::CountryStructure, period::EvalPeriod) -> BitVector
    eval_periods(cs::CountryStructure, ::CompletePeriod) -> UnitRange
Returns a mask or a range of indices of the periods included in
`EvalPeriod` or `CompletePeriod` to apply *slicing* to the inflation
trajectories and the parameter before obtaining the evaluation metrics.

See also: [`EvalPeriod`](@ref), [`CompletePeriod`](@ref), [`period_tag`](@ref).
"""
function eval_periods end

# Function to return the mask of periods to evaluate, with respect to the
# inflation periods of a CountryStructure
function eval_periods(cs::CountryStructure, period::EvalPeriod)
    dates = infl_dates(cs)
    period.startdate .<= dates .<= period.finaldate
end

function eval_periods(cs::CountryStructure, ::CompletePeriod)
    1:infl_periods(cs)
end

# Label for results
"""
    period_tag(period::EvalPeriod) -> String
    period_tag(::CompletePeriod) -> String
Function to obtain the label associated with the evaluation period. The
complete evaluation period has an empty label (`""`).

See also: [`EvalPeriod`](@ref), [`CompletePeriod`](@ref), [`eval_periods`](@ref).
"""
period_tag(period::EvalPeriod) = period.tag
period_tag(::CompletePeriod) = "" 

# Extend Base.string to print period
Base.show(io::IO, ::CompletePeriod) = print(io, "Complete period")
Base.show(io::IO, p::EvalPeriod) = print(io, p.tag * ":" * Dates.format(p.startdate, DEFAULT_DATE_FORMAT) * "-" * Dates.format(p.finaldate, DEFAULT_DATE_FORMAT))

# Definition of default periods for evaluation of Guatemala data
"""
    const GT_EVAL_B00 = EvalPeriod(Date(2001, 12), Date(2010, 12), "gt_b00")
Default period for evaluation in the decade of the 2000s, including the year 2010.
"""
const GT_EVAL_B00 = EvalPeriod(Date(2001, 12), Date(2010, 12), "gt_b00")

"""
    const GT_EVAL_B10 = EvalPeriod(Date(2011, 12), Date(2021, 12), "gt_b10")
Default period for evaluation in the decade of the 2010s, including the year 2021.
"""
const GT_EVAL_B10 = EvalPeriod(Date(2011, 12), Date(2023, 12), "gt_b10")

"""
    const GT_EVAL_T0010 = EvalPeriod(Date(2011, 1), Date(2011, 11), "gt_t0010")
Default period for evaluation in the transition from the 2000s to the 2010s.
"""
const GT_EVAL_T0010 = EvalPeriod(Date(2011, 1), Date(2011, 11), "gt_t0010")



## Iteration over periods

# These methods are defined for functions involving CrossEvalConfig with a
# period or a collection of periods
Base.length(::EvalPeriod) = 1
Base.size(e::EvalPeriod) = (1,)
Base.iterate(e::EvalPeriod) = e, nothing
Base.iterate(::EvalPeriod, ::Nothing) = nothing


##########################################################################################
### CHANGES 15/11/2023 by DJGM

struct PeriodVector <: AbstractEvalPeriod
    V::Vector{Tuple{Date,Date}}  
    tag::String 
end


function eval_periods(cs::CountryStructure, v::PeriodVector)
    dates = infl_dates(cs)
    .|([period[1] .<= dates .<= period[2] for period in v.V]...)
end

period_tag(period::PeriodVector) = period.tag