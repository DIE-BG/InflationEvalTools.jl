using Dates, CPIDataBase, InflationFunctions, InflationEvalTools
using CPIDataBase.TestHelpers
using CPIDataBase: index_dates
using CPIDataGT
using Test
using Random

# Load CPI data
CPIDataGT.load_data()
test_GTDATA23 = GTDATA23

@testset "ResampleExtendedSVM tests" begin

    # Testing ResampleExtendedSVM
    svmext_sampler = ResampleExtendedSVM([150, 180, 50])
    resample_extended_data = svmext_sampler(test_GTDATA23)

    # Create a function to check if the all elements of a vector are contains in another one
    chekingMembership = (x::Vector, y::Vector) -> x .∈ y

    # I think there is only one basic expenditure that needs to be checked.
    # If you see zeros in the result, it means that there are no common elements
    # from one basic expense in the others.
    # Check the Januaries in CPI base 2000
    result = mapreduce(vcat, Iterators.product(1:12, 1:12)) do (i, j)
        i == j ? 0 :
            chekingMembership(
                resample_extended_data[1].v[i:12:end, 1],
                [test_GTDATA23[1].v[j:12:end, 1]]
            )
    end

    @test sum(result) == 0 # there ar no common elements between months

    # Check the Januaries in CPI base 2010
    result = mapreduce(vcat, Iterators.product(1:12, 1:12)) do (i, j)
        i == j ? 0 :
            chekingMembership(
                resample_extended_data[2].v[i:12:end, 1],
                [test_GTDATA23[2].v[j:12:end, 1]]
            )
    end

    @test sum(result) == 0 # there ar no common elements between months

    # Check the Januaries in CPI base 2023
    result = mapreduce(vcat, Iterators.product(1:12, 1:12)) do (i, j)
        i == j ? 0 :
            chekingMembership(
                resample_extended_data[3].v[i:12:end, 1],
                [test_GTDATA23[3].v[j:12:end, 1]]
            )
    end

    @test sum(result) == 0 # there ar no common elements between months

end

@testset "Testing defining a single extension period" begin
    # The final number of peridios for all VarCPIBase must be 150
    svmext_sampler = ResampleExtendedSVM(150)
    resample_extended_data = svmext_sampler(test_GTDATA23)

    @test size(resample_extended_data[1].v, 1) == 150
    @test size(resample_extended_data[2].v, 1) == 150
    @test size(resample_extended_data[3].v, 1) == 150
    # Check dates
    @test periods(resample_extended_data) == length(index_dates(resample_extended_data))
    @test all(map(base -> issorted(base.dates), resample_extended_data.base))

end

@testset "Testing for VarCPIBase types" begin
    # We provide a VarCPIBase like a intup
    svmext_sampler = ResampleExtendedSVM(150)
    resample_extended_data = svmext_sampler(GT00)

    @test isa(resample_extended_data, VarCPIBase)
    # Check dates
    @test periods(resample_extended_data) == length(index_dates(resample_extended_data))

end

## Add a testset for computing the population dataset with extended VarCPIBase objects
@testset "Population datasets with extended VarCPIBase objects" begin

    svmext_sampler = ResampleExtendedSVM(150)
    param_fn = get_param_function(svmext_sampler)
    population_dataset = param_fn(test_GTDATA23)
    # Test
    @test all(map(periods, population_dataset.base) .== 150)

    svmext_sampler = ResampleExtendedSVM([150, 180, 50])
    param_fn = get_param_function(svmext_sampler)
    population_dataset = param_fn(test_GTDATA23)
    @test all(map(periods, population_dataset.base) .== svmext_sampler.extension_periods)
end
