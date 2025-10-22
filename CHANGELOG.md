# Change Log

All notable changes to this project will be documented in this file.
 
## [Unreleased] - 2025-10
### Added 
- Type `CPIVarietyMatchDistribution` to resample as in the B-TIMA extension methodology. It uses two arrays of monthly price changes, a prior and an actual empirical distribution. It can be configured to sample from the prior distribution, the actual  distribution or a synthtetic distribution that gives different weights according to the mean of the actual observations.
- Resampling function `ResampleSynthetic` to implement an interface to resample `VarCPIBase` objects.
- Resampling function `ResampleIdentity` to resample `CountryStructure`s and `VarCPIBase` objects. 
- Resampling function `ResampleMixture` to implement an ensemble of samplers for the `VarCPIBase` components. This allows setting mixed resampling schemes for different CPI datasets. For example, in a `CountryStructure` with two `VarCPIBase`s, not resampling the first dataset (`ResampleIdentity`) and using the same calendar months for the second CPI dataset (`ResampleScrambleVarMonths`).

### Changed
- Refactored `evalsim` into `compute_lowlevel_sim`. 
- Refactored `makesim` into `compute_assessment_sim`.
- Refactored `run_batch` into `run_assessment_batch`.
- Make concrete type `PeriodVector` public.
- Removed `ResampleSBB`, `ResampleGSBB` and `ResampleGSBBMod`.
- Removed `CrossEvalConfig` type and methods.

## [0.2.1] - 2025-09
### Fixed
- Improved documentation of assessment functions and types. 

## [0.2.0] - 2025-07
### Added 
- Internal package became public.