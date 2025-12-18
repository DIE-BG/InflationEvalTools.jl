```@meta
CurrentModule = InflationEvalTools
```

# InflationEvalTools

Types, functions, and simulation utilities for the evaluation of inflation measures.

```@docs
InflationEvalTools.InflationEvalTools
```

## Defaults

```@docs
DEFAULT_SEED
LOCAL_RNG
DEFAULT_RESAMPLE_FN
DEFAULT_TREND_FN
```

## Resampling Methods

Resampling functions for CPI bases and related utilities.

```@docs
ResampleFunction
ResampleScrambleVarMonths
ResampleScrambleTrended
ResampleTrended
ResampleExtendedSVM
ResampleIdentity
get_param_function
method_name
method_tag
```

## B-TIMA Extension 

The B-TIMA extension procedure uses the following functions: 

```@docs
CPIVarietyMatchDistribution
ResampleSynthetic
synthetic_reweighing
prior_reweighing
actual_reweighing
ResampleMixture
```

## Trend Functions

Functions for trend application and modeling.

```@docs
RWTREND
TrendFunction
ArrayTrendFunction
TrendRandomWalk
get_ranges
TrendAnalytical
TrendExponential
TrendIdentity
TrendDynamicRW
zeromean_validation
create_TrendDynamicRW_array
```

## Parametric Base Methods

Methods to obtain bases of population monthly price changes.

```@docs
AbstractInflationParameter
param_scramblevar_fn
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
SimDynamicConfig
AbstractEvalPeriod
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
gentrajinfl
pargentrajinfl
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
_merge_metrics
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
share_combination_weights_rmse
share_combination_weights_absme
```