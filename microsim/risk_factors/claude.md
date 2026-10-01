# Risk Factors Module Documentation

This document provides detailed guidance for working with risk factors in the MICROSIM framework. For general architecture and project overview, see `../../CLAUDE.md`.

## Overview

The risk_factors module manages both static (demographic) and dynamic (time-varying) risk factors that drive individual health trajectories in the microsimulation. Risk factors are modeled using statistical models trained on NHANES and Kaiser Permanente data.

**Key concepts:**
- **Static risk factors**: Demographics that don't change over time (race, education, gender, smoking status, modality)
- **Dynamic risk factors**: Time-varying physiological and behavioral factors (age, blood pressure, cholesterol, BMI, etc.)
- **Risk models**: Statistical models (linear, logistic, Cox) that predict how risk factors evolve over time

## Risk Factor Types

### Static Risk Factors

Static risk factors are set at Person initialization and remain constant throughout the simulation.

**Available static risk factors:**
- `gender`: Male/Female (see `gender.py`)
- `raceEthnicity`: Racial/ethnic categories (see `race_ethnicity.py`)
- `education`: Educational attainment levels (see `education.py`, `education_model.py`)
- `smokingStatus`: Smoking status categories (see `smoking_status.py`)
- `modality`: MRI modality indicator for neuroimaging studies (see `modality.py`, `modality_model.py`)

### Dynamic Risk Factors

Dynamic risk factors evolve over time according to statistical models and are updated at each simulation wave.

**Available dynamic risk factors:**
- **Age**: `age` (see `age_model.py`)
- **Blood pressure**: `sbp` (systolic), `dbp` (diastolic)
- **Metabolic**: `a1c` (glycated hemoglobin), `bmi` (body mass index)
- **Lipids**: `hdl`, `ldl`, `trig` (triglycerides), `totChol` (total cholesterol)
- **Cardiovascular**: `afib` (atrial fibrillation, see `afib_model.py`), `pvd` (peripheral vascular disease, see `pvd_model.py`)
- **Behavioral**: `anyPhysicalActivity`, `alcoholPerWeek` (see `alcohol_model.py`, `alcohol_category.py`)
- **Anthropometric**: `waist` (waist circumference, see `waist_model.py`)
- **Renal**: `creatinine`

### Risk Factor Categories

Risk factors are categorized as either:
- **Categorical**: Discrete categories (e.g., gender, race, education, smoking status)
- **Continuous**: Numeric values (e.g., age, blood pressure, cholesterol, BMI)

This distinction affects how they are modeled and used in outcome predictions. See `risk_factor.py` for the full enumeration and category definitions (`CategoricalRiskFactorsType`, `ContinuousRiskFactorsType`).

## Model Implementations

### Model Types

The framework supports multiple statistical model types for risk factor prediction:

1. **Linear Models**: `LinearRiskFactorModel`
   - For continuous outcomes with normal distributions
   - Used for: Blood pressure, cholesterol, BMI, etc.

2. **Logistic Models**: `LogisticRiskFactorModel`
   - For binary outcomes
   - Used for: Afib, PVD, smoking status transitions

3. **Cox Proportional Hazards**: `CoxRiskFactorModel`
   - For time-to-event outcomes
   - Used for: First occurrence of conditions

4. **Specialized Models** (live in `risk_factors/`, not `regression_models/`):
   - `NHANESLinearRiskFactorModel` (`nhanes_linear_risk_factor_model.py`): legacy NHANES pickle-based linear model
   - `LogLinearRiskFactorModel` (`log_linear_risk_factor_model.py`): log-transformed variant of `NHANESLinearRiskFactorModel`

`LinearRiskFactorModel`, `LogisticRiskFactorModel`, and `CoxRiskFactorModel` are in `microsim/regression_models/`.

### Model Specification Files

Risk factor model parameters (regression coefficients, intercepts, variance) are stored in JSON files:
- Location: `data/*CohortModelSpec.json`
- Contains: Regression coefficients and model parameters for various risk factors (LDL, BMI, A1C, etc.)
- Format: Cohort-specific (NHANES vs Kaiser)

These JSON specifications are loaded by the model repositories and used to parameterize the statistical models.

## Key Files in This Module

- `risk_factor.py`: Core enumerations (DynamicRiskFactorsType, StaticRiskFactorsType, CategoricalRiskFactorsType, ContinuousRiskFactorsType)
- `risk_model_repository.py`: Base `RiskModelRepository` with bounds enforcement and model initialization helpers
- `cohort_risk_model_repository.py`: Cohort-specific repositories (`CohortDynamicRiskFactorModelRepository`, `CohortStaticRiskFactorModelRepository`, and the helper `AlcoholCategoryModel`)
- `initialization_model_repository.py`: `InitializationModelRepository` — seeds PVD, AFIB, WAIST, EDUCATION, ALCOHOL, MODALITY at Person construction
- Individual model files:
  - `age_model.py`: Age progression
  - `afib_model.py`: Atrial fibrillation
  - `pvd_model.py`: Peripheral vascular disease
  - `waist_model.py`: Waist circumference
  - `alcohol_model.py`: Alcohol consumption
  - `education_model.py`: Education (initialization)
  - `modality_model.py`: MRI modality (initialization)
  - `nhanes_linear_risk_factor_model.py`: `NHANESLinearRiskFactorModel` — NHANES-specific linear model (legacy pickle-based)
  - `log_linear_risk_factor_model.py`: `LogLinearRiskFactorModel` — log-transformed variant of `NHANESLinearRiskFactorModel`
  - `a1c.py`: Utility functions to convert between fasting glucose and A1C (`convert_fasting_glucose_to_a1c`, `convert_a1c_to_fasting_glucose`)
  - `gfr_equation.py`: `GFREquation` — CKD-EPI glomerular filtration rate calculation
- Demographic enums:
  - `gender.py`: Gender categories
  - `race_ethnicity.py`: Race/ethnicity categories
  - `smoking_status.py`: Smoking status categories
  - `education.py`: Education level categories
  - `alcohol_category.py`: Alcohol consumption categories

## Integration with Person Class

Risk factors are stored in Person instances as:

```python
# In person.py
_staticRiskFactors: dict
    # Demographics that don't change (race, education, gender, smoking status, modality)
    # Set during Person initialization via InitializationRepository

_dynamicRiskFactors: dict[int, dict]
    # Time-varying factors (age, blood pressure, BMI, cholesterol, etc.)
    # Keyed by wave number, updated each simulation advance
```

When a Person is advanced in time, dynamic risk factors are updated using the models from `RiskModelRepository` (accessed via `PopulationRepositoryType.DYNAMIC_RISK_FACTORS`).

## Repository Pattern

Risk factors follow the Repository Pattern:

```
RiskModelRepository (risk_model_repository.py)  ← base class
  ↓ subclassed by
CohortDynamicRiskFactorModelRepository (cohort_risk_model_repository.py)
  ↓ loads from
data/*CohortModelSpec.json (via load_regression_model)
```

Repositories are accessed through the Population's `PopulationRepositoryType.DYNAMIC_RISK_FACTORS` or `PopulationRepositoryType.STATIC_RISK_FACTORS` enum.

## Modifying Risk Factor Models

### Updating Model Parameters

To modify risk factor model coefficients or parameters:

1. **Edit cohort model specifications**:
   - Locate the appropriate JSON file in `data/*CohortModelSpec.json`
   - Update regression coefficients, intercepts, or variance parameters
   - Ensure JSON structure matches expected schema

2. **Modify model implementations**:
   - Edit `*_risk_factor_model.py` files in `microsim/regression_models/`
   - Update model logic, transformations, or estimation methods

3. **Update risk factor enumerations**:
   - Add/remove risk factors in `risk_factor.py`
   - Update `DynamicRiskFactorsType` or `StaticRiskFactorsType` enums
   - Specify category (Categorical vs Continuous)

4. **Update repositories**:
   - Modify `risk_model_repository.py` to register new models
   - Update `cohort_risk_model_repository.py` for cohort-specific configurations

### Adding a New Risk Factor

To add a completely new risk factor:

1. Add enum entry to `risk_factor.py` (either `DynamicRiskFactorsType` or `StaticRiskFactorsType`)
2. Create model file if needed (e.g., `new_risk_factor_model.py`)
3. Add model specification to `data/*CohortModelSpec.json` if using statistical model
4. Register model in appropriate repository (`risk_model_repository.py` or initialization repository)
5. Update `person.py` to track the new risk factor
6. Update `model_argument_transform.py` if the risk factor is used in outcome models

## Common Tasks

### Accessing Risk Factors from a Person

```python
# Static risk factors
gender = person._staticRiskFactors[StaticRiskFactorsType.GENDER]
race = person._staticRiskFactors[StaticRiskFactorsType.RACE_ETHNICITY]

# Dynamic risk factors (at specific wave)
wave = 5
sbp = person._dynamicRiskFactors[wave][DynamicRiskFactorsType.SBP]
age = person._dynamicRiskFactors[wave][DynamicRiskFactorsType.AGE]
```

### Using Risk Factors in Outcome Models

Risk factors are automatically transformed into model arguments using `model_argument_transform.py`. This ensures Person attributes are correctly formatted for outcome model predictions.

### Testing Risk Factor Models

Risk factor tests are found in `test/test_risk_model_repository.py`, `test/test_linear_risk_factor_model.py`, `test/test_nhanes_linear_risk_factor_model.py`, and related files. Common test patterns:

```python
import unittest
from microsim.population import PopulationFactory

class TestRiskFactorModel(unittest.TestCase):
    def setUp(self):
        self.pop = PopulationFactory.get_nhanes_population(n=100, year=1999)

    def test_risk_factor_progression(self):
        # Test that risk factors evolve correctly over time
        self.pop.advance(years=1)
        # Assert expected behavior
```

## Cross-References

- **Main architecture**: `../../CLAUDE.md`
- **Trial framework**: `../trials/claude.md`
- **Outcome models**: `../outcomes/` (outcomes depend on risk factors for predictions)
- **Treatment strategies**: `../treatment_strategies/` (treatments modify risk factors)
- **Person class**: `../person.py` (risk factor storage and evolution)
- **Model argument transformation**: `../regression_models/model_argument_transform.py` (risk factor usage in models)
