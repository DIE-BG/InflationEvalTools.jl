```@meta
CurrentModule = InflationEvalTools
```

# InflationEvalTools

Types, functions, and simulation utilities for the evaluation of inflation measures.

## Resampling Methods

Resampling functions for CPI bases and related utilities.

```@docs
ResampleScrambleVarMonths
ResampleScrambleTrended
ResampleTrended
get_param_function
method_name
method_tag
```

## Trend Functions

Functions for trend application and modeling.

```@docs
RWTREND
TrendRandomWalk
TrendAnalytical
TrendExponential
TrendIdentity
```

## Parametric Base Methods

Methods to obtain bases of population monthly price changes.

```@docs
param_gsbb_mod
param_sbb
InflationParameter
ParamTotalCPIRebase
ParamTotalCPI
ParamWeightedMean
ParamTotalCPILegacyRebase
```

## Simulation Configuration

Types and utilities for simulation and evaluation period configuration.

```@docs
AbstractConfig
SimConfig
CompletePeriod
EvalPeriod
PeriodVector
eval_periods
period_tag
GT_EVAL_B00
GT_EVAL_B10
GT_EVAL_T0010
```

## Trajectory Generation

Functions for generating inflation trajectories.

```@docs
gentrayinfl
pargentrayinfl
```

## Evaluation & Metrics

Functions for simulation evaluation and metrics.

```@docs
compute_lowlevel_sim
compute_assessment_sim
dict2config
run_assessment_batch
eval_metrics
combination_metrics
eval_mse_online
eval_absme_online
eval_corr_online
```

## Combination Methods

Optimal combination of estimators and related utilities.

```@docs
combination_weights
average_mats
ridge_combination_weights
lasso_combination_weights
share_combination_weights
elastic_combination_weights
metric_combination_weights
absme_combination_weights
```

## Cross-Validation

Functions for cross-validation and related utilities.

```@docs
add_ones
crossvalidate
```
