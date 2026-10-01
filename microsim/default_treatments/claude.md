# Default Treatments Module Documentation

For general architecture and project overview, see `../../CLAUDE.md`.

## Overview

The default_treatments module manages **usual care** treatments that persons receive as part of standard medical practice, separate from experimental treatment strategies evaluated in trials. Default treatments represent baseline medical management and evolve over time based on person characteristics and risk factors.

**Key characteristics:**
- **Baseline treatments**: Represent usual care, not experimental interventions
- **Time-varying**: Default treatments can change at each simulation wave based on clinical guidelines and person characteristics
- **Repository-based**: Managed via `DefaultTreatmentRepository` following the same repository pattern as risk factors
- **Modifiable by strategies**: Treatment strategies can override default treatments during simulation

Default treatments integrate with the Person class through:
- `Person._defaultTreatments`: List of default treatment types being tracked
- Treatment values stored as time-indexed arrays (like dynamic risk factors)
- Applied during `person.advance()` before treatment strategies

## Default Treatment Types

From `default_treatments/default_treatments.py`:

```python
class DefaultTreatmentsType(Enum):
    STATIN = "statin"
    ANTI_HYPERTENSIVE_COUNT = "antiHypertensiveCount"
```

### Treatment Categories

Default treatments are categorized similarly to risk factors:

**Categorical Treatments** (`CategoricalDefaultTreatmentsType`):
- `STATIN`: Statin therapy status (binary: on/off)

**Continuous Treatments** (`ContinuousDefaultTreatmentsType`):
- `ANTI_HYPERTENSIVE_COUNT`: Number of antihypertensive medications (integer count)


## Key Files in This Module

### Core Files
- `default_treatments.py`: Default treatment enumerations (DefaultTreatmentsType, CategoricalDefaultTreatmentsType, ContinuousDefaultTreatmentsType)

### Repository Implementation
- `default_treatment_model_repository.py`: Contains `DefaultTreatmentModelRepository`
  - Registers statistical models for predicting default treatment changes over time
  - Uses linear probability models and integer-rounded linear models

### Model Specification Files
- Location: `data/*CohortModelSpec.json`
- Examples: `statinCohortModel`, `antiHypertensiveCountCohortModel`
- Format: Regression coefficients for predicting treatment changes based on person characteristics

## Architecture: Repository Pattern

Default treatments use the same repository pattern as risk factors and outcomes:

```
DefaultTreatmentModelRepository
  ↓ extends
RiskModelRepository
  ↓ contains
Statistical models for each default treatment type
  ↓ loaded from
data/*CohortModelSpec.json
```

**Repository hierarchy:**
1. `DefaultTreatmentModelRepository` (in `default_treatments/default_treatment_model_repository.py`):
   - Inherits from `RiskModelRepository`
   - Maps DefaultTreatmentsType → Statistical model
   - Initializes models using cohort model specifications

2. **Model types**:
   - `LinearRiskFactorModel`: For probability-based treatments (statin)
   - `RoundedLinearRiskFactorModel`: For count-based treatments (medication counts)

## Integration with Person Class

Default treatments are tracked in Person instances similar to dynamic risk factors:

```python
# In person/person.py (lines 91-93)
for key, value in defaultTreatmentsDict.items():
    setattr(self, "_" + key, [value])
self._defaultTreatments = list(defaultTreatmentsDict.keys())
```

**Access patterns:**
- `person._statin`: Array of statin status by wave
- `person._antiHypertensiveCount`: Array of antihypertensive medication counts by wave
- `person.get_last_default_treatment(DefaultTreatmentsType.STATIN.value)`: Get current treatment value (takes a string, not an enum)

**Wave semantics:**
- Default treatments stored as time-indexed arrays (like dynamic risk factors)
- Wave -1: Before first advance (initialized from baseline data)
- Wave 0+: Updated each simulation year

## Simulation Flow: Default Treatments

During `person.advance()` (person/person.py:102-129), default treatments are processed in this order:

1. **Dynamic risk factors updated** (`advance_risk_factors()`)
2. **Default treatments updated** (`advance_treatments()`) ← Default treatments applied here
3. **Treatment strategies applied** (`advance_treatment_strategies_and_update_risk_factors()`)
   - Treatment strategies can **override** default treatments
4. **Outcomes evaluated** (`advance_outcomes()`)

### Treatment Update Process

From person/person.py:147-154:

```python
def advance_treatments(self, defaultTreatmentRepository):
    """Makes predictions for the default treatments 1 year to the future."""
    for treatment in self._defaultTreatments:
        setattr(self, "_" + treatment,
                getattr(self, "_" + treatment) +
                [self.get_next_treatment(treatment, defaultTreatmentRepository)])

def get_next_treatment(self, treatment, treatmentRepository):
    model = treatmentRepository.get_model(treatment)
    return model.estimate_next_risk(self)
```

### Treatment Strategy Override

Treatment strategies can modify default treatments via person/person.py:169-173:

```python
def update_treatments(self, treatmentStrategy):
    updatedTreatments = treatmentStrategy.get_updated_treatments(self)
    for treatment in self._defaultTreatments:
        if treatment in updatedTreatments.keys():
            getattr(self, "_" + treatment)[-1] = updatedTreatments[treatment]
```

## Repository Access

Default treatment repository is accessed via `PopulationRepositoryType.DEFAULT_TREATMENTS`:

```python
# From population_model_repository.py
class PopulationRepositoryType(Enum):
    STATIC_RISK_FACTORS = "staticRiskFactors"
    DYNAMIC_RISK_FACTORS = "dynamicRiskFactors"
    DEFAULT_TREATMENTS = "defaultTreatments"  # Default treatment repository
    OUTCOMES = "outcomes"

# In PopulationModelRepository.__init__:
self._repository = {
    PopulationRepositoryType.DEFAULT_TREATMENTS.value: defaultTreatmentRepository,
    ...
}
```

## Common Tasks

### Accessing Default Treatments from a Person

```python
# Get current (most recent) treatment value (method takes the enum's .value string)
current_statin = person.get_last_default_treatment(DefaultTreatmentsType.STATIN.value)
current_bp_meds = person.get_last_default_treatment(DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value)

# Access full treatment history
statin_history = person._statin  # Array indexed by wave
bp_med_history = person._antiHypertensiveCount  # Array indexed by wave

# Access at specific wave
wave = 5
statin_at_wave_5 = person._statin[wave]
```

### Using Default Treatments in Outcome Models

Default treatments are available in outcome model predictions via `model_argument_transform.py`. For example, statin status affects cardiovascular outcome risk.

### Modifying Default Treatment Logic

**To update treatment prediction models:**
1. Edit cohort model specification JSON in `data/` directory (e.g., `statinCohortModel.json`)
2. Modify regression coefficients to reflect new clinical guidelines or evidence
3. Reload the model (or restart simulation)

**To add a new default treatment type:**

1. **Add enum to `default_treatments.py`**:
   ```python
   class DefaultTreatmentsType(Enum):
       STATIN = "statin"
       ANTI_HYPERTENSIVE_COUNT = "antiHypertensiveCount"
       NEW_TREATMENT = "newTreatment"  # Add here
   ```

2. **Categorize the treatment**:
   ```python
   # Add to appropriate category enum
   class CategoricalDefaultTreatmentsType(Enum):
       STATIN = "statin"
       NEW_TREATMENT = "newTreatment"  # If categorical

   # OR

   class ContinuousDefaultTreatmentsType(Enum):
       ANTI_HYPERTENSIVE_COUNT = "antiHypertensiveCount"
       NEW_TREATMENT = "newTreatment"  # If continuous
   ```

3. **Create model specification**:
   - Add JSON spec file: `data/newTreatmentCohortModel.json`
   - Include regression coefficients for predicting treatment use

4. **Register in repository** (`default_treatments/default_treatment_model_repository.py`):
   ```python
   class DefaultTreatmentModelRepository(RiskModelRepository):
       def __init__(self):
           super().__init__()
           self._initialize_linear_probability_risk_model(
               DefaultTreatmentsType.NEW_TREATMENT.value,
               "newTreatmentCohortModel"
           )
   ```

5. **Update Person initialization**:
   - Ensure new treatment is initialized in `person_factory.py`
   - Add to default treatments dictionary during person creation

6. **Update treatment strategies** (if applicable):
   - Modify treatment strategies in `treatment_strategies/` to potentially override the new default treatment

## Interaction with Treatment Strategies

**Critical distinction:**
- **Default treatments**: Usual care, baseline medical management
- **Treatment strategies**: Experimental interventions being evaluated in trials

**Relationship:**
- Default treatments are applied first during `person.advance()`
- Treatment strategies can **override** default treatments in the same wave
- When strategies override, the last value in the treatment array is replaced
- This allows trials to compare experimental strategies against usual care baseline

**Example:**
```python
# Default treatment predicts person receives 2 BP medications
person._antiHypertensiveCount = [1, 2]  # Wave -1, Wave 0

# Treatment strategy overrides to intensive BP control (4 medications)
treatmentStrategy.get_updated_treatments(person)  # Returns {antiHypertensiveCount: 4}
person.update_treatments(treatmentStrategy)
person._antiHypertensiveCount = [1, 4]  # Override applied to Wave 0
```

For detailed treatment strategy documentation, see `microsim/treatment_strategies/claude.md`.

## Testing Patterns

Default treatment testing typically involves:
- Verifying correct model initialization and coefficient loading
- Testing treatment progression over time
- Ensuring treatment strategies correctly override defaults
- Validating treatment counts and probabilities are within valid ranges

**Example test structure:**
```python
import unittest
from microsim.population import PopulationFactory
from microsim.default_treatments.default_treatments import DefaultTreatmentsType

class TestDefaultTreatments(unittest.TestCase):
    def setUp(self):
        self.pop = PopulationFactory.get_nhanes_population(n=100, year=1999)

    def test_default_treatment_initialization(self):
        # Test that persons have initial default treatment values
        for person in self.pop._people:
            self.assertIsNotNone(person._statin)
            self.assertIsNotNone(person._antiHypertensiveCount)

    def test_default_treatment_progression(self):
        # Test that default treatments update over time
        self.pop.advance(years=1)
        for person in self.pop._people:
            self.assertEqual(len(person._statin), 2)  # Initial + 1 year
```

## Cross-References

- **Main CLAUDE.md**: For overall architecture, Person/Population structure, wave semantics
- **microsim/risk_factors/claude.md**: Default treatments use same repository pattern and model types
- **microsim/treatment_strategies/claude.md**: For experimental treatments that override defaults
- **microsim/outcomes/claude.md**: Outcome models may use default treatments as predictors
- **person/person.py**: For default treatment storage, tracking, and update logic (lines 91-93, 147-154, 169-173)
- **population_model_repository.py**: For repository access via PopulationRepositoryType enum
- **default_treatments/default_treatment_model_repository.py**: For DefaultTreatmentModelRepository implementation
