using Dates, CPIDataBase, InflationFunctions
using CPIDataBase.TestHelpers
using CPIDataGT
using Test
using Random
using TerminalPager
include("BTIMA-ResampleExtendedSVM_Helpers.jl")

# Load CPI data
CPIDataGT.load_data()

# Testing ResampleExtendedSVM
resampleExtfn = ResampleExtendedSVM([150, 180, 50])
ResampleExtended = resampleExtfn(GTDATA23)

# Create a function to check if the all elements of a vector are contains in another one
checking_membership = (x::Vector, y::Vector) -> x .∈ y

# I think there is only one basic expenditure that needs to be checked.
# If you see zeros in the result, it means that there are no common elements 
# from one basic expense in the others.
# Check the Januaries in CPI base 2000
result = mapreduce(vcat,Iterators.product(1:12, 1:12)) do (i,j)
    i == j ? 0 : 
            checking_membership(
                ResampleExtended[1].v[i:12:end, 1], 
                [GTDATA23[1].v[j:12:end, 1]])
end

sum(result) # there ar no common elements between months

# Check the Januaries in CPI base 2010
result = mapreduce(vcat,Iterators.product(1:12, 1:12)) do (i,j)
    i == j ? 0 : 
            checking_membership(
                ResampleExtended[2].v[i:12:end, 1], 
                [GTDATA23[2].v[j:12:end, 1]])
end

sum(result) # there ar no common elements between months

# Check the Januaries in CPI base 2010
result = mapreduce(vcat,Iterators.product(1:12, 1:12)) do (i,j)
    i == j ? 0 : 
            checking_membership(
                ResampleExtended[3].v[i:12:end, 1], 
                [GTDATA23[3].v[j:12:end, 1]])
end

sum(result) # there ar no common elements between months


# Check the Januaries in CPI base 2000













