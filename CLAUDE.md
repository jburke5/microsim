# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MICROSIM is an open-source chronic disease microsimulation framework for population health modeling. It simulates individual cardiovascular risk factors, health outcomes, and cognition at the person level, using data from NHANES (National Health and Nutrition Examination Survey) and Kaiser Permanente cohorts.

**Key characteristics:**
- Individual-level agent-based simulation (Person → Population → Trial hierarchy)
- Models 20+ risk factors (both static and dynamic) and 11+ health outcomes
- Supports clinical trial simulations with treatment strategy comparisons
- Python 3.12+ with NumPy, Pandas, Statsmodels, and Lifelines for statistical modeling

## Development Commands

### Setup
```bash
poetry install  # Install dependencies in virtualenv
```

### Testing
```bash
poetry run test  # Run all tests with unittest (discovers test_*.py in test/)
python -m unittest microsim.test.test_<module_name>  # Run specific test module
```

### Code Quality
```bash
poetry run lint    # Check code with flake8
poetry run format  # Format code with black (line-length: 99)
```

## Architecture Overview

### Directory Structure

The codebase is organized into functional directories:
- `common/`: Shared enums and data-access helpers (`AgeScope`, `PopulationType`, `VariableType`, `data_loader`) - See `microsim/common/claude.md` for detailed documentation
- `person/`: The `Person` agent plus factory and filter utilities - See `microsim/person/claude.md` for detailed documentation
- `population/`: `Population`, `PopulationFactory`, and the population model repositories - See `microsim/population/claude.md` for detailed documentation
- `regression_models/`: Reusable regression/statistical model building blocks shared by risk factors and outcomes - See `microsim/regression_models/claude.md` for detailed documentation
- `outcomes/`: All outcome models and repositories - See `microsim/outcomes/claude.md` for detailed documentation
- `risk_factors/`: Risk factor models and enums (age, BP, cholesterol, demographics, etc.) - See `microsim/risk_factors/claude.md` for detailed documentation
- `treatment_strategies/`: Treatment strategy definitions - See `microsim/treatment_strategies/claude.md` for detailed documentation
- `default_treatments/`: Default treatment enums and application logic - See `microsim/default_treatments/claude.md` for detailed documentation
- `trials/`: Trial framework (orchestration, analysis, outcome assessment) - See `microsim/trials/claude.md` for detailed documentation
- `test/`: Unit tests with fixtures and helpers - See `microsim/test/claude.md` for test guidelines
- `data/`: NHANES and Kaiser datasets, model specifications

### Core Design Pattern: Repository + Factory

The codebase uses **Repository Pattern** to manage collections of models and **Factory Pattern** to construct complex objects:

```
Person (individual agent)
  ↓ created by
PersonFactory
  ↓ uses data from
fullyImputedDataset.dta (NHANES) or Kaiser data

Population (collection of Persons)
  ↓ created by
PopulationFactory
  ↓ configured with
PopulationModelRepository (sub-repositories keyed by PopulationRepositoryType)
  ├── CohortDynamicRiskFactorModelRepository (dynamic risk factors)
  ├── CohortStaticRiskFactorModelRepository  (static risk factors)
  ├── DefaultTreatmentModelRepository        (usual-care treatments)
  └── OutcomeModelRepository                 (health outcomes)

(TreatmentStrategyRepository is NOT part of PopulationModelRepository. Experimental
 treatment strategies are injected separately via Population.advance(..., treatmentStrategies=...)
 / Trial.run(), layered on top of the default usual-care models above.)

Trial (experimental design)
  ↓ compares
Multiple Populations with different TreatmentStrategies
  ↓ analyzed by
TrialOutcomeAssessor (Cox regression, logistic regression, linear regression, relative risk, incidence rate)

  See microsim/trials/claude.md for detailed trial framework documentation
```

### Person and Wave Semantics

**Critical concept:** A "wave" represents a time transition in the simulation.
- Wave numbering starts at -1 before first advance
- Wave 1 = transition from subscript[0] → subscript[1]
- If a person has an event during wave 1: status is Negative at [0], Positive at [1]
- See the `Person` class docstring in `person/person.py` for detailed explanation

**Person structure:**
- `_staticRiskFactors`: Demographics that don't change (race, education, gender)
- `_dynamicRiskFactors`: Time-varying factors (age, blood pressure, BMI, cholesterol)
- `_defaultTreatments`: Usual care treatments
- `_treatmentStrategies`: Experimental treatment strategies
- `_outcomes`: Dictionary of outcome arrays, keyed by OutcomeType
- `_randomEffects`: Outcome model random effects storage

**Advancing simulation:**
```python
person.advance(years, dynamicRiskFactorRepository, defaultTreatmentRepository,
               outcomeModelRepository, treatmentStrategies)
```

### Population Structure

Populations manage collections of Person instances with two execution modes:
- `advance_serial()`: Single-threaded execution
- `advance_parallel()`: Multi-process execution with `nWorkers` parameter

**Key repositories (accessed via PopulationRepositoryType enum):**
- `STATIC_RISK_FACTORS`: Initialization models
- `DYNAMIC_RISK_FACTORS`: Risk factor progression models
- `DEFAULT_TREATMENTS`: Treatment application rules
- `OUTCOMES`: Outcome prediction models

### Model Types and Implementations

**Risk Factor Models:**
See `microsim/risk_factors/claude.md` for detailed documentation on risk factor model types (Linear, Logistic, Cox, specialized models) and implementations.

**Outcome Models:**
See `microsim/outcomes/claude.md` for detailed outcome model documentation.

**Model Specification Files:**
Located in `data/*CohortModelSpec.json` - these JSON files contain regression coefficients and model parameters for various risk factors (LDL, BMI, A1C, etc.)

### Treatment Strategies

See `microsim/treatment_strategies/claude.md` for detailed treatment strategy documentation.

### Trial Framework

The `trials/` directory contains the experimental design framework for simulating clinical trials and comparing treatment strategies.

**For detailed trial framework documentation, see `microsim/trials/claude.md`**

## Data Sources

**Primary datasets:**
- `data/fullyImputedDataset.dta`: NHANES-based population data (Stata format)
- `data/kaiser/*.csv`: Kaiser Permanente cohort data (mean, covariance, weights, min/max)
- `data/us.1969_2017.19ages.adjusted.txt`: US mortality tables by age/year

**Population Types:**
The framework supports two population sources via `population_type.py`:
- NHANES: US representative sample
- Kaiser: California-based cohort (different risk profiles)

## Key Files for Common Tasks

### Creating a new outcome model
See `microsim/outcomes/claude.md` for detailed guidance on creating and modifying outcome models.

### Modifying risk factor models
See `microsim/risk_factors/claude.md` for detailed guidance on modifying risk factor models, including cohort model specifications, model implementations, enums, and repositories.

### Creating a new treatment strategy
See `microsim/treatment_strategies/claude.md` for detailed guidance on creating and modifying treatment strategies.

### Running a population simulation
1. Create population: Use `PopulationFactory.get_nhanes_population()` or `get_kaiser_population()`
2. Define treatment strategies (optional): Configure via `TreatmentStrategiesType` or use predefined strategies
3. Run simulation: Call `population.advance(years)` to advance the population forward in time
4. Analyze results: Use population reporting methods to extract outcomes and statistics

**For clinical trial simulations with multiple arms and statistical comparisons, see `microsim/trials/claude.md`**

## Important Enumerations

**Risk Factors** (`risk_factors/risk_factor.py`):
- Static: `gender`, `raceEthnicity`, `education`, `smokingStatus`, `modality`
- Dynamic: `age`, `sbp`, `dbp`, `a1c`, `hdl`, `ldl`, `trig`, `totChol`, `bmi`, `physicalActivity`, `afib`, `waist`, `alcoholPerWeek`, `creatinine`, `pvd`
- Categories: Categorical vs Continuous
- **See `microsim/risk_factors/claude.md` for detailed risk factor documentation**

**Treatments** (`default_treatments/default_treatments.py`, `treatment_strategies/treatment_strategies.py`):
- Default: `antiHypertensiveCount`, `statin`
- **See `microsim/default_treatments/claude.md` for detailed default treatment documentation**
- Strategies: Defined in `treatment_strategies/` directory per protocol
- **See `microsim/treatment_strategies/claude.md` for detailed treatment strategy documentation**

**Outcomes** (`outcomes/outcome.py`):
- `STROKE`, `MI`, `DEMENTIA`, `DEATH`, `CARDIOVASCULAR_DEATH`, `NON_CARDIOVASCULAR_DEATH`
- `COGNITIVE_IMPAIRMENT`, `WMH`, `EPILEPSY`, `QALY`
- **See `microsim/outcomes/claude.md` for detailed outcome documentation**

## Testing Patterns

Tests follow unittest framework conventions:
- All tests in `test/test_*.py`
- Use `TestCase` base class
- Helper utilities in `test/helper/` (e.g., `init_vectorized_population_dataframe.py`)
- Test coverage includes: outcome models, risk factors, population reporting, and trial operations (see `microsim/trials/claude.md` for trial testing details)

**Typical test structure:**
```python
class TestPopulation(unittest.TestCase):
    def setUp(self):
        # Tests typically create populations directly using PopulationFactory
        self.pop = PopulationFactory.get_nhanes_population(
            n=100, year=1999, personFilters=None,
            nhanesWeights=True, distributions=False
        )
```

## Code Style

- **Formatter:** Black with 99-character line length
- **Linter:** Flake8
- **Naming:** Snake_case for variables/functions, PascalCase for classes
- **Private attributes:** Prefix with underscore (e.g., `_waveCompleted`)

## Common Gotchas

1. **Path handling:** Data files use `get_absolute_datafile_path()` from `data_loader.py` to resolve paths correctly. It anchors on the package location, so it works from any working directory; a literal `"microsim/data/..."` only resolves when the process runs from the repository root
2. **Wave indexing:** Remember waves are 0-indexed after first advance (-1 before any advance)
3. **Repository access:** Population repositories accessed via `PopulationRepositoryType` enum, not direct dictionary keys
4. **Random number generation:** Each Person has its own RNG (`_rng`) for reproducibility in multiprocessing
5. **Model arguments:** Risk factor models use `model_argument_transform.py` to convert Person attributes to model inputs
