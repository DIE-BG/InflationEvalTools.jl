"""
    abstract type TrendFunction <: Function end
Abstract type to handle trend functions.

## Usage
    function (trendfn::TrendFunction)(cs::CountryStructure)
Applies the trend function to a `CountryStructure` and returns a new
`CountryStructure`.
"""
abstract type TrendFunction <: Function end


""" 
    abstract type ArrayTrendFunction <: TrendFunction end
Type for trend function that stores the vector of values to apply to
the month-to-month variations.

## Usage 

    function (trendfn::ArrayTrendFunction)(base::VarCPIBase{T}, range::UnitRange) where T
Specifies how to apply the trend function to a VarCPIBase with the index
range `range`.
"""
abstract type ArrayTrendFunction <: TrendFunction end


"""
    method_name(resamplefn::TrendFunction)
Function to obtain the name of the trend function.
"""
method_name(::TrendFunction) = error("The name of the trend function must be redefined")

## Implementation of general behavior for trend application function

"""
    method_tag(trendfn::TrendFunction)
Function to obtain a tag for the trend function.
"""
method_tag(trendfn::TrendFunction) = string(nameof(trendfn))


"""
    get_ranges(cs::CountryStructure)
Helper function to obtain a tuple of index ranges for slicing the
trend vectors.
"""
function get_ranges(cs::CountryStructure) 
    # Get the periods of each base
    periods = map( base -> size(base.v, 1), cs.base)
    # Generate a vector of ranges and fill each range with the indices
    # formed with the elements of periods
    ranges = Vector{UnitRange}(undef, length(periods))
    start = 0
    for i in eachindex(periods)
        ranges[i] = start + 1 : start + periods[i]
        start = periods[i]
    end
    # Return a tuple of index ranges
    NTuple{length(cs.base), UnitRange{Int64}}(ranges)
end

# General application of TrendFunction to CountryStructure
function (trendfn::TrendFunction)(cs::CountryStructure)
    # Get index ranges for the bases of the CountryStructure
    ranges = get_ranges(cs)
    # Apply the trend function to each base. It is required to define for
    # any TrendFunction how to operate on the tuple (::VarCPIBase,
    # ::UnitRange)
    newbases = map(trendfn, cs.base, ranges)
    # Build a new CountryStructure with the modified bases
    typeof(cs)(newbases)
end


## Implementation of trend application for ArrayTrendFunction

# Application of ArrayTrendFunction, which stores the trend vector to be
# applied to a VarCPIBase
function (trendfn::ArrayTrendFunction)(base::VarCPIBase{T}, range::UnitRange) where T
    # Get the trend vector from the trend field
    trend::Vector{T} = @view trendfn.trend[range]
    # Conditionally apply the trend to the month-to-month variation matrix
    vtrend =  @. base.v * ((base.v > 0) * trend + !(base.v > 0))
    # Create a new VarCPIBase with the new month-to-month variations
    VarCPIBase(vtrend, base.w, base.dates, base.baseindex)
end 


## Definition of type for random walk trend

"""
    TrendRandomWalk{T} <: ArrayTrendFunction

Type to represent a random walk trend function. Uses the
precalibrated random walk vector in [`RWTREND`](@ref).

# Example: 
```julia-repl 
# Create the random walk trend function
trendfn = TrendRandomWalk()
```
"""
Base.@kwdef struct TrendRandomWalk{T} <: ArrayTrendFunction
    trend::Vector{T} = RWTREND
end

# Name for the random walk trend function
method_name(::TrendRandomWalk) = "Random walk trend"
method_tag(::TrendRandomWalk) = "RW"

## Definition of type for analytical (anonymous function) trend

"""
    TrendAnalytical{T} <: ArrayTrendFunction

Type to represent a trend function defined by an anonymous function.
Receives the data from a `CountryStructure` or a range of indices to precompute
the trend vector using an anonymous function.

## Examples: 

To create a trend function from an anonymous function: 
```julia-repl 
trendfn = TrendAnalytical(param_data, t -> 1 + sin(2π*t/12), "Sinusoidal trend")
```
or: 
```julia-repl 
trendfn = TrendAnalytical(1:periods(param_data), t -> 1 + sin(2π*t/12), "Sinusoidal trend")
```
"""
struct TrendAnalytical{T} <: ArrayTrendFunction
    trend::Vector{T}
    name::String

    # Constructor method to obtain the number of periods to compute from a CountryStructure
    function TrendAnalytical(cs::CountryStructure, fnhandle::Function, name::String)
        # Get the number of periods from the bases of the CountryStructure
        p = periods(cs)
        # Create a vector with the function mapped over the periods
        trend::Vector{eltype(cs)} = fnhandle.(1:p)
        # Return with the same type as the CountryStructure used.
        new{eltype(cs)}(trend, name)
    end
    # Constructor method from a range of periods
    function TrendAnalytical(range::UnitRange, fnhandle::Function, name::String)
        # Map a function over the elements of a UnitRange 
        trend::Vector{Float32} = fnhandle.(range)
        # Return as Float32
        new{Float32}(trend, name)
    end

end

# Name for analytical trend function, must be provided in the constructor
method_name(trendfn::TrendAnalytical) = trendfn.name
method_tag(::TrendAnalytical) = "TA" # depends on the vector generated by the anonymous function

## Definition of type for identity (neutral) trend function

"""
    TrendIdentity <: TrendFunction

Concrete type to represent a neutral trend function. That is, this
trend function leaves the data unchanged. 

## Examples: 
```julia-repl 
# Create an identity trend function. 
trendfn = TrendIdentity()
```

## Usage 
    function (trendfn::TrendIdentity)(cs::CountryStructure)

Application of TrendIdentity trend to VarCPIBase. This method is redefined
to leave the VarCPIBase unchanged. 
```julia-repl 
trendfn = TrendIdentity() 
trended_cs = trendfn(gtdata) 
```
"""
struct TrendIdentity <: TrendFunction end

# Name for the identity trend function
method_name(::TrendIdentity) = "Identity trend"
method_tag(::TrendIdentity) = "ID"

# Redefine to return the same CountryStructure unchanged
function (trendfn::TrendIdentity)(cs::CountryStructure)
    # Simply return the CountryStructure
    cs
end 



## Definition of type for exponential trend

# This trend function applies an exponential growth model with the
# specified rate

"""
    TrendExponential{T} <: ArrayTrendFunction

Concrete type to represent an exponential growth trend function. 

## Constructors 
    function TrendExponential(cs::CountryStructure, rate::Real = 0.02f0)
    function TrendExponential(range::UnitRange, rate::Real = 0.02f0)

## Examples: 
```julia-repl 
# Create a trend function with 2% annual exponential growth
trendfn = TrendExponential(gtdata, 0.02)
```

## Usage 
    function (trendfn::TrendExponential)(cs::CountryStructure)

Application of TrendExponential trend to the `VarCPIBase` objects that
make up the `CountryStructure`. 
```julia-repl 
trendfn = TrendExponential(gtdata, 0.02) 
trended_cs = trendfn(gtdata) 
```
"""
struct TrendExponential{T} <: ArrayTrendFunction 
    trend::Vector{T}
    rate::Float32

    # Constructor method to obtain the number of periods to compute from a CountryStructure
    function TrendExponential(cs::CountryStructure, rate::Real = 0.02f0)

        rate > 1 && error("Growth rate must be less than one. ")
        
        # Get the number of periods from the bases of the CountryStructure
        p = periods(cs)
        # Create a vector with the function mapped over the periods
        frate = Float32(rate)
        fnhandle = t -> ((1 + frate)^(1/12)) ^ t
        trend::Vector{eltype(cs)} = fnhandle.(1:p)
        # Return with the same type as the CountryStructure used.
        new{eltype(cs)}(trend, frate)
    end
    
    # Constructor method from a range of periods
    function TrendExponential(range::UnitRange, rate::Real = 0.02f0)
        
        rate > 1 && error("Growth rate must be less than one.")

        # Create a vector with the function mapped over the periods
        frate = Float32(rate)
        fnhandle = t -> ((1 + frate)^(1/12)) ^ t
        # Map a function over the elements of a UnitRange 
        trend::Vector{Float32} = fnhandle.(range)
        # Return as Float32
        new{Float32}(trend, frate)
    end
end

# Name for the exponential growth trend function
method_name(trendfn::TrendExponential) = "Exponential growth trend at " * string(round(100 * trendfn.rate, digits=2)) * "%"
method_tag(trendfn::TrendExponential) = "EXP" * string(round(100 * trendfn.rate, digits=2)) * "%"


