using Dates, CPIDataBase, InflationFunctions, InflationEvalTools
using CPIDataBase.TestHelpers
using CPIDataGT
using Test
using Random

# Load CPI data
CPIDataGT.load_data()

@testset "Test of ResampleExtendedSVM" begin

    # Testing ResampleExtendedSVM
    resampleExtfn = ResampleExtendedSVM([150, 180, 50])
    resample_extended_data = resampleExtfn(GTDATA23)

    # Create a function to check if the all elements of a vector are contains in another one
    chekingMembership = (x::Vector, y::Vector) -> x .∈ y

    # I think there is only one basic expenditure that needs to be checked.
    # If you see zeros in the result, it means that there are no common elements 
    # from one basic expense in the others.
    # Check the Januaries in CPI base 2000
    result = mapreduce(vcat,Iterators.product(1:12, 1:12)) do (i,j)
        i == j ? 0 : 
                chekingMembership(
                    resample_extended_data[1].v[i:12:end, 1], 
                    [GTDATA23[1].v[j:12:end, 1]])
    end

    @test sum(result) == 0 # there ar no common elements between months

    # Check the Januaries in CPI base 2010
    result = mapreduce(vcat,Iterators.product(1:12, 1:12)) do (i,j)
        i == j ? 0 : 
                chekingMembership(
                    resample_extended_data[2].v[i:12:end, 1], 
                    [GTDATA23[2].v[j:12:end, 1]])
    end

    @test sum(result) == 0 # there ar no common elements between months

    # Check the Januaries in CPI base 2023
    result = mapreduce(vcat,Iterators.product(1:12, 1:12)) do (i,j)
        i == j ? 0 : 
                chekingMembership(
                    resample_extended_data[3].v[i:12:end, 1], 
                    [GTDATA23[3].v[j:12:end, 1]])
    end

    @test sum(result) == 0 # there ar no common elements between months

end

@testset "Testing defining a single extension period" begin
    # The final number of peridios for all VarCPIBase must be 150
    resampleExtfn = ResampleExtendedSVM(150)
    resample_extended_data = resampleExtfn(GTDATA23)

    @test size(resample_extended_data[1].v, 1) == 150
    @test size(resample_extended_data[2].v, 1) == 150
    @test size(resample_extended_data[3].v, 1) == 150        

end

@testset "Testing for VarCPIBase types" begin
    # We provide a VarCPIBase like a intup
    resampleExtfn = ResampleExtendedSVM(150)
    resample_extended_data = resampleExtfn(GT00)

    @test isa(resample_extended_data, VarCPIBase)
 
end