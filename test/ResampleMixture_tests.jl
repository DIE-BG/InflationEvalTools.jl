##  ----------------------------------------------------------------------------
#   Test script for the ResampleMixture type
#   ----------------------------------------------------------------------------
using InflationEvalTools
using Test
using CPIDataBase: index_dates
using CPIDataGT
using Distributions: pdf, Normal
using Statistics: mean, std
using StatsBase: sample, sample!
import Random
import Random: AbstractRNG

# Load GT data
CPIDataGT.load_data()
# Uncomment the line below to run this script manually
# include("BTIMA_extension_helpers.jl")  # Extension helper functions

##  ----------------------------------------------------------------------------
#   Setup test data and helper samplers
#   ----------------------------------------------------------------------------

# Test data - Get a subset of bases for testing
GTDATA_test = GTDATA23[Date(2001, 1), Date(2025, 9)]

# Create test samplers

# 1. Identity sampler (no resampling)
identity_sampler = ResampleIdentity()

# 2. Synthetic sampler (from BTIMA extension)
# Setup matching array for synthetic sampler
code_b10 = "_0111101"
code_b23 = "_0111101"
m = 1
v_10, v_23 = get_obs_month(code_b10, code_b23, m)
var1 = CPIVarietyMatchDistribution(v_10, v_23, m)

GT23_test = GTDATA_test[3]
nperiods = periods(GT23_test)
nitems = items(GT23_test)
matching_array = [var1 for m in 1:12, j in 1:nitems]
synthetic_sampler = ResampleSynthetic(GT23_test, matching_array)

# 3. ScrambleVarMonths sampler
scramblevar_sampler = ResampleScrambleVarMonths()

##  ----------------------------------------------------------------------------
#   Test ResampleMixture functionality
#   ----------------------------------------------------------------------------

@testset "ResampleMixture tests" begin

    @testset "ResampleMixture: Constructor and basic properties" begin
        # Create mixture sampler
        mixture_sampler = ResampleMixture([identity_sampler, scramblevar_sampler, synthetic_sampler])

        @test mixture_sampler isa InflationEvalTools.ResampleFunction
        @test mixture_sampler.resampling_functions[1] === identity_sampler
        @test mixture_sampler.resampling_functions[2] === scramblevar_sampler
        @test mixture_sampler.resampling_functions[3] === synthetic_sampler
        @test method_name(mixture_sampler) == "Mixture of Resampling Methods"
        @test method_tag(mixture_sampler) == "MIX"
    end


    @testset "ResampleMixture: Validation" begin
        # Should fail with wrong number of samplers
        @test_throws ErrorException ResampleMixture([identity_sampler])(GTDATA_test)
        @test_throws ErrorException ResampleMixture([identity_sampler, scramblevar_sampler])(GTDATA_test)
    end


    @testset "ResampleMixture: Resampling CountryStructure" begin
        mixture_sampler = ResampleMixture([identity_sampler, scramblevar_sampler, synthetic_sampler])

        # Test with default RNG
        resampled_cs = mixture_sampler(GTDATA_test)
        @test resampled_cs isa typeof(GTDATA_test)
        @test length(resampled_cs.base) == length(GTDATA_test.base)

        # First base should be identical (identity sampler)
        @test resampled_cs.base[1] === GTDATA_test.base[1]

        # Second base should be resampled (scramblevar sampler)
        @test resampled_cs.base[2] !== GTDATA_test.base[2]
        # Third base should be resampled (synthetic sampler)
        @test resampled_cs.base[3] !== GTDATA_test.base[3]

        # Check dates
        @test length(index_dates(resampled_cs)) == periods(resampled_cs)
        @test all(map(base -> issorted(base.dates), resampled_cs.base))

        # Test with specific RNG
        rng = Random.MersenneTwister(1234)
        resampled_cs_seeded = mixture_sampler(GTDATA_test, rng)
        @test resampled_cs_seeded isa typeof(GTDATA_test)
    end


    @testset "ResampleMixture: Resampling VarCPIBase" begin
        mixture_sampler = ResampleMixture([identity_sampler, synthetic_sampler])

        # Should use first sampler (identity) when applied to VarCPIBase
        resampled_base = mixture_sampler(GT23_test)
        @test resampled_base === GT23_test  # Identity sampler returns same object

        # Test with specific RNG
        rng = Random.MersenneTwister(1234)
        resampled_base_seeded = mixture_sampler(GT23_test, rng)
        @test resampled_base_seeded === GT23_test
    end


    @testset "ResampleMixture: Parameter functions" begin
        synthetic_sampler = ResampleSynthetic(GT23_test, matching_array)
        mixture_sampler = ResampleMixture([identity_sampler, scramblevar_sampler, synthetic_sampler])
        param_fn = get_param_function(mixture_sampler)

        # Get parameter data
        param_cs = param_fn(GTDATA_test)
        @test param_cs isa typeof(GTDATA_test)
        @test length(param_cs.base) == length(GTDATA_test.base)

        # First base should be identical (from identity sampler)
        @test param_cs.base[1] === GTDATA_test.base[1]

        # Second base should be the population data from the scramblevar sampler
        pop_varbase_scramble = InflationEvalTools.param_scramblevar_fn(GTDATA_test.base[2])
        @test param_cs.base[2] !== GTDATA_test.base[2]
        @test param_cs.base[2].v == pop_varbase_scramble.v
        @test periods(param_cs.base[2]) == periods(GTDATA_test.base[2])

        # Third base should be the population data from the synthetic sampler
        pop_vmat_synthetic_fn = get_param_function(synthetic_sampler)
        pop_varbase_synthetic = pop_vmat_synthetic_fn(GTDATA_test.base[3])
        @test param_cs.base[3] !== GTDATA_test.base[3]      # Not equal to the actual data
        @test param_cs.base[3].v == pop_varbase_synthetic.v     # Equal to the population data
        @test periods(param_cs.base[3]) == periods(GTDATA_test.base[3])
        # Check dates
        @test length(index_dates(param_cs)) == periods(param_cs)
        @test all(map(base -> issorted(base.dates), param_cs.base))

        # Test parameter function validation
        @test_throws ErrorException param_fn(GTDATA24)
    end


    # Test the ResampleMixture works well when extending some of the VarCPIBase objects through the samplers
    @testset "ResampleMixture: Extension of VarCPIBase objects" begin

        # Last resampler samples more periods
        synthetic_sampler = ResampleSynthetic(GT23_test, matching_array, 12 * 5)
        mixture_sampler = ResampleMixture([identity_sampler, scramblevar_sampler, synthetic_sampler])
        resampled_cs = mixture_sampler(GTDATA_test)

        # More periods in the bootstrap sample
        @test periods(resampled_cs) > periods(GTDATA_test)
        # First base should be identical (identity sampler)
        @test resampled_cs.base[1] === GTDATA_test.base[1]
        # Second base should be resampled (scramblevar sampler)
        @test resampled_cs.base[2] !== GTDATA_test.base[2]
        # Third base should be resampled (synthetic sampler)
        @test resampled_cs.base[3] !== GTDATA_test.base[3]
        @test size(resampled_cs.base[3].v, 1) > size(GTDATA_test.base[3].v, 1)
        @test size(resampled_cs.base[3].v, 2) == size(GTDATA_test.base[3].v, 2)
        # Check dates
        @test length(index_dates(resampled_cs)) == periods(resampled_cs)
        @test all(map(base -> issorted(base.dates), resampled_cs.base))

        ## Some tests combining with ResampleExtendedSVM
        svmext_sampler = ResampleExtendedSVM(180)
        mixture_sampler = ResampleMixture([identity_sampler, svmext_sampler, synthetic_sampler])
        resampled_cs = mixture_sampler(GTDATA_test)

        # Check extension of periods
        @test periods(resampled_cs.base[2]) == 180
        @test periods(resampled_cs.base[3]) == 12 * 5

        # Check dates
        @test length(index_dates(resampled_cs)) == periods(resampled_cs)
        @test all(map(base -> issorted(base.dates), resampled_cs.base))

    end

end
