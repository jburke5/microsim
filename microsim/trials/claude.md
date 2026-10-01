# Trial Framework Documentation

This document provides detailed guidance for working with the trial simulation framework in MICROSIM.

## Overview

The trial framework enables simulation-based clinical trial comparisons, allowing researchers to test different treatment strategies against control populations. Trials use statistical analysis methods (Cox regression, logistic regression, linear regression, relative risk, incidence rate) to assess outcomes and compare interventions.

## Directory Structure

The `trials/` directory contains the experimental design framework:
- `trial.py`: Main trial orchestration, `__str__` results printout, and CSV export (`export_results`)
- `trial_description.py`: Configuration for trial setup
- `trial_type.py`: Trial type enumeration
- `trial_outcome_assessor.py`: Analysis and results computation
- `trial_outcome_assessor_factory.py`: Factory for creating outcome assessors
- `trial_factory.py`: One-shot helpers (`TrialFactory.run_nhanes`, `TrialFactory.run_kaiser`) that build the description, run the trial, and analyze it with a default-or-supplied assessor in a single call
- Regression analysis modules:
  - `cox_regression_analysis.py`: Cox proportional hazards analysis
  - `logistic_regression_analysis.py`: Logistic regression for binary outcomes
  - `linear_regression_analysis.py`: Linear regression for continuous outcomes
  - `relative_risk_analysis.py`: Relative risk calculations
  - `incidence_rate_analysis.py`: Incidence rate analysis (events per 1000 person-years)
  - `regression_analysis.py`: Base regression analysis class

## Trial Components

### Trial Orchestration

**trial.py** - The main Trial class that:
- Manages multiple populations (treatment arms)
- Coordinates simulation execution across all arms
- Delegates outcome assessment to TrialOutcomeAssessor

**trial_description.py** - Configuration object that specifies:
- Trial design parameters
- Population characteristics
- Treatment strategy assignments
- Analysis methods to apply

**trial_type.py** - Enumeration of supported trial types

### Outcome Assessment

**TrialOutcomeAssessor** - Analyzes trial results using:
- **Cox regression**: Time-to-event analysis with hazard ratios
- **Logistic regression**: Binary outcome analysis with odds ratios
- **Linear regression**: Continuous outcome analysis
- **Relative risk analysis**: Direct risk comparisons between arms
- **Incidence rate analysis**: Events per 1000 person-years for each arm

The assessor compares populations and generates statistical summaries of treatment effects.

## Running a Trial Simulation

### Quick Path: TrialFactory

For the common case (build a description, run, analyze with the default assessor),
use `TrialFactory.run_nhanes` or `TrialFactory.run_kaiser`. Each is a single call
that returns a completed, analyzed `Trial`:

```python
from microsim.trials.trial_factory import TrialFactory

trial = TrialFactory.run_nhanes(sampleSize=1000, duration=5,
                                treatmentStrategies="1bpMedsAdded")
print(trial)  # formatted results table
```

Pass `exportPath="trial-results.csv"` to also write the results to a CSV file laid out like the
printout: the trial description once at the top, then one block per analysis type with a header row
and one row per assessment. The default `exportPath=None` writes nothing.

`treatmentStrategies` accepts the same forms as `TrialDescription` (None, a shorthand
string, or a `TreatmentStrategyRepository`). Pass `assessor=` to override the default
`TrialOutcomeAssessor`. All population-specific kwargs (`year`, `nhanesWeights`,
`distributions`, `prevalenceRiskScaling` for NHANES; `wmhSpecific`, `riskScaling` for
Kaiser) are forwarded to the corresponding description subclass.

Note that `NhanesTrialDescription` defaults `nhanesWeights=True`, so an NHANES trial samples
with the survey weights unless you ask it not to. `year` defaults to 1999. The description
passes `nhanesWeights` on explicitly, so it does not inherit `PopulationFactory`'s behaviour
of letting an unset `nhanesWeights` follow `distributions` — a trial is weighted either way.

For finer control — inspecting populations before `run()`, swapping in a custom
assessor between `run()` and `analyze()`, or skipping analysis entirely — use the
step-by-step process below.

### Step-by-Step Process

The Trial constructs its own treated and control populations internally from the
TrialDescription — you do not create populations and pass them in.

1. **Define treatment strategies** (optional): Configure via `TreatmentStrategiesType`,
   pass a shorthand string, or build a `TreatmentStrategyRepository` directly. `None`
   yields an empty repository (control-only style usage).
   ```python
   from microsim.treatment_strategies.treatment_strategy_repository import TreatmentStrategyRepository
   treatmentStrategies = TreatmentStrategyRepository.from_string("...")
   ```

2. **Build a population-specific TrialDescription**: Use `NhanesTrialDescription` or
   `KaiserTrialDescription`. The base `TrialDescription` is an abstract class
   (`abc.ABC` with an abstract `__init__`) and cannot be instantiated directly.
   The description carries sample size, duration, trial type,
   randomization/block factors, person filters, worker count, and population-specific
   args (e.g. NHANES year, weights; Kaiser wmhSpecific, riskScaling).
   ```python
   from microsim.trials.trial_description import NhanesTrialDescription
   from microsim.trials.trial_type import TrialType

   description = NhanesTrialDescription(
       trialType=TrialType.COMPLETELY_RANDOMIZED,
       sampleSize=1000,
       duration=5,
       treatmentStrategies=treatmentStrategies,
       year=1999,
   )
   ```

3. **Instantiate the Trial**: Construction itself draws people from the
   `PopulationFactory` and assembles the treated and control `Population` objects
   (`trial.treatedPop`, `trial.controlPop`) via `get_trial_populations()`.
   ```python
   from microsim.trials.trial import Trial
   trial = Trial(description)
   ```

4. **Run the simulation**: `trial.run()` advances both arms — control for the full
   duration with no treatment strategies, treated for 1 wave with the strategy's
   initial (`BEGIN`) status, after which `run()` flips each active strategy to
   `TreatmentStrategyStatus.MAINTAIN` and advances the treated arm the remaining
   `duration-1` waves. (The status enum has only `BEGIN`, `MAINTAIN`, `END`.)
   ```python
   trial.run()
   ```

5. **Analyze results**: Build a `TrialOutcomeAssessor` describing which outcomes and
   analyses to run, then call `trial.analyze(assessor)` (or `trial.run_analyze(assessor)`
   to do both in one step). Results are stored in `trial.results`, keyed by
   `AnalysisType.value` then by assessment name.
   ```python
   from microsim.trials.trial_outcome_assessor_factory import TrialOutcomeAssessorFactory
   assessor = TrialOutcomeAssessorFactory.get_trial_outcome_assessor(...)
   trial.analyze(assessor)
   ```
   `trial.run_analyze(assessor, exportPath=...)` runs, analyzes, and exports in one call;
   `trial.export_results(path)` writes the CSV for an already-analyzed trial and
   `trial.get_results_dfs()` returns the same blocks as a dict of DataFrames keyed by analysis type.

## Testing Trial Components

Trial-related tests are located in `test/test_*.py` and include:
- Trial orchestration tests
- Outcome assessment validation
- Regression analysis verification
- Population comparison tests

Tests use the standard unittest framework.

## Integration with Core Framework

Trials sit at the top of the MICROSIM hierarchy:

```
Trial (experimental design)
  ↓ compares
Multiple Populations with different TreatmentStrategies
  ↓ analyzed by
TrialOutcomeAssessor (Cox regression, logistic regression, linear regression, relative risk, incidence rate)
```

Each trial arm is a complete Population with its own:
- PersonFactory initialization
- TreatmentStrategyRepository
- Outcome tracking and reporting

## Common Trial Patterns

### Comparing Treatment Strategies

Create multiple populations with different treatment strategies, run them in parallel, and compare outcomes:
- Control arm: Standard of care (no additional treatment)
- Treatment arm(s): Specific intervention strategies

### Analyzing Multiple Outcomes

TrialOutcomeAssessor can analyze multiple outcome types simultaneously:
- Cardiovascular outcomes (STROKE, MI, CARDIOVASCULAR_DEATH)
- Cognitive outcomes (DEMENTIA, COGNITIVE_IMPAIRMENT)
- Overall mortality (DEATH)

### Statistical Methods Selection

Choose analysis method based on outcome type:
- **Time-to-event data**: Cox regression (most common for clinical trials)
- **Binary outcomes at fixed timepoint**: Logistic regression
- **Continuous outcomes**: Linear regression
- **Simple risk comparisons**: Relative risk analysis
- **Incidence rates per 1000 person-years**: Incidence rate analysis

## Adding a New Analysis Type

When adding a new `AnalysisType` to the outcome assessor, **always update all of the following**:

1. `trial_outcome_assessor.py` — add the enum value and register the class in `ANALYSIS_CLASSES` (`_analysis` is built from it)
2. `trial_outcome_assessor.py` — update validation in `add_outcome_assessment` if the new type requires a non-standard set of assessment functions (`cox` uses `{"outcome", "time"}`, `incidenceRate` uses `{"eventAndTime"}` pair functions, the rest use `{"outcome"}`)
3. `incidence_rate_analysis.py` (or new file) — implement the analysis class with an `analyze(trial, assessmentFunctionDict, assessmentAnalysis)` method and a `columns` class attribute naming each element of the returned tuple, in order (`get_results_dfs` zips them strictly, so a mismatch raises)
4. `trial_outcome_assessor_factory.py` — add default assessments for the new type
5. **`trial.py` `__str__` method** — add an `elif analysisType == AnalysisType.NEW_TYPE:` branch with an appropriate column header for the results printout
6. `claude.md` (this file) — update the statistical methods list and directory structure
