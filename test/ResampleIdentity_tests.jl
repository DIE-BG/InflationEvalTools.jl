using Test
using InflationEvalTools
using CPIDataGT
using Random

# Load GT data used elsewhere in tests
CPIDataGT.load_data()

# Small helper: a VarCPIBase from GT24
base = GT24

resamp = ResampleIdentity()

@test method_name(resamp) == "Identity Resampling (no-op)"
@test method_tag(resamp) == "IDTY"

# Resampling a VarCPIBase should return the same object
sampled_base = resamp(base)
@test sampled_base === base

# Resampling a CountryStructure should return the same object
sampled_cs = resamp(GTDATA24)
@test sampled_cs === GTDATA24

# get_param_function should return a function that yields the same VarCPIBase
popfn = get_param_function(resamp)
returned = popfn(base)
@test returned === base

