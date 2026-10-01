# Outcomes Module Documentation

For general architecture and project overview, see `../../CLAUDE.md`.

## Overview

The outcomes module models health events that occur during the microsimulation. Each outcome has:
- A type (defined in OutcomeType enum)
- Fatality status (fatal or non-fatal)
- Optional phenotype details (NIHSS score for stroke, QALY values, etc.)

Outcomes integrate with the Person class through:
- `Person._outcomes`: Dictionary keyed by OutcomeType, storing outcome arrays by wave
- `Person._randomEffects`: Stores outcome model random effects for consistency across waves
- Wave-based tracking: Outcomes occur during specific waves and are indexed accordingly

## Outcome Types and Ordering

**CRITICAL:** Outcome evaluation order matters because some outcomes depend on others.

From `outcomes/outcome.py`:
```python
class OutcomeType(Enum):
    WMH = "wmh"                                    # White matter hyperintensities (first)
    COGNITION = "cognition"                        # Cognitive function
    CI = "ci"                                      # Cognitive impairment
    MCI = "mci"                                    # Mild cognitive impairment
    DIABETES = "diabetes"                          # Diabetes
    CHRONIC_KIDNEY_DISEASE = "chronicKidneyDisease" # Chronic kidney disease
    CARDIOVASCULAR = "cv"                          # General CV event
    STROKE = "stroke"                              # Stroke (partitioned from CV)
    MI = "mi"                                      # Myocardial infarction (partitioned from CV)
    NONCARDIOVASCULAR = "noncv"                    # Non-CV outcomes
    DEMENTIA = "dementia"                          # Dementia (after cognition)
    EPILEPSY = "epilepsy"                          # Epilepsy
    DEATH = "death"                                # Death (near end)
    QUALITYADJUSTED_LIFE_YEARS = "qalys"           # QALYs (last)
```

`outcome.py` also defines `EventOutcomeType` (same members as `OutcomeType` minus `COGNITION`), used when only discrete events need to be tracked.

**Dependency rationale:**
- Cognition evaluated before stroke (stroke affects cognition)
- Cognition before CI/MCI/dementia (these derive from cognition)
- CV evaluated before stroke/MI (partition models)
- NonCV after CV/stroke/MI (to determine cause of death)
- Death before QALYs (QALYs need death status)

## Key Files in This Module

### Core Infrastructure
- `outcome.py`: Outcome class, OutcomeType enum, architectural guidance (lines 3-13 have HOW TO add outcome)
- `outcome_model_repository.py`: Central registry mapping OutcomeType → ModelRepository

### Cardiovascular Outcomes
- `ascvd_outcome_model.py`: ASCVD risk prediction
- `cv_model.py`: General cardiovascular outcome model
- `cv_model_repository.py`: CV model repository

### Stroke Models
- `stroke_partition_model.py`: Stroke probability model (partitions from CV events)
- `stroke_partition_model_repository.py`: Stroke model repository (gender-specific)
- `stroke_outcome.py`: StrokeOutcome class with phenotype (NIHSS, subtype, type)
- `stroke_details.py`: Stroke subtype and type models

### MI Models
- `mi_partition_model.py`: MI probability model (partitions from CV events)
- `mi_partition_model_repository.py`: MI model repository (gender-specific)

### Dementia and Cognition
- `dementia_model.py`: Dementia outcome model
- `dementia_model_repository.py`: Dementia model repository
- `cognition_outcome.py`: Cognition outcome class
- `cognition_model.py`: GCP (Global Cognitive Performance) models — `GCPModel` (general) and `GCPStrokeModel` (post-stroke)
- `cognition_model_repository.py`: Cognition model repository (selects between `GCPModel` and `GCPStrokeModel`)
- `ci_model.py`: Cognitive impairment model
- `ci_model_repository.py`: CI model repository
- `mci_model.py`: Mild cognitive impairment model
- `mci_model_repository.py`: MCI model repository

### White Matter Hyperintensities (WMH)
- `wmh_model.py`: WMH outcome model
- `wmh_model_repository.py`: WMH model repository
- `wmh_presence_model.py`: WMH presence prediction
- `wmh_outcome.py`: WMH outcome class
- `wmh_severity.py`: WMH severity enumeration
- `wmh_severity_unknown.py`: Unknown WMH severity handling

### Epilepsy
- `epilepsy_model.py`: Epilepsy outcome model
- `epilepsy_model_repository.py`: Epilepsy model repository

### Death
- `death_model.py`: Death outcome model
- `death_model_repository.py`: Death model repository
- `non_cv_death_model.py`: Non-cardiovascular death model
- `non_cv_model_repository.py`: Non-CV model repository

### QALY
- `qaly_outcome.py`: QALY outcome class
- `qaly_model_repository.py`: QALY model repository
- `qaly_assignment_strategy.py`: QALY assignment strategies

### Diabetes
- `diabetes_model.py`: Diabetes outcome model (`DiabetesModel`) and prevalence seeding model (`DiabetesPrevalenceModel`); gated on A1C ≥ 6.5 for first detection
- `diabetes_model_repository.py`: Diabetes model repository (`DiabetesModelRepository`, `DiabetesPrevalenceModelRepository`)

### Chronic Kidney Disease
- `chronic_kidney_disease_model.py`: CKD outcome model (`ChronicKidneyDiseaseModel`) and prevalence seeding model (`ChronicKidneyDiseasePrevalenceModel`); gated on GFR < 60 for first detection
- `chronic_kidney_disease_model_repository.py`: CKD model repository (`ChronicKidneyDiseaseModelRepository`, `ChronicKidneyDiseasePrevalenceModelRepository`)

### Outcome Prevalence Infrastructure
- `outcome_prevalence_base.py`: Abstract base class `OutcomePrevalenceBase` for all priorToSim prevalence models; subclasses implement `get_linear_predictor_for_person` and set `_outcomeType`
- `outcome_prevalence_model_repository.py`: `OutcomePrevalenceModelRepository` — parallel to `OutcomeModelRepository` but for priorToSim seeding; also holds `DEFAULT_PREVALENCE_RISK_SCALING` dict with calibrated per-outcome risk scaling values

### Other Models
- `sbi_model.py`: SBI (Silent Brain Infarction) model
- `reference.py`: `Reference` class holding external epidemiological reference data (e.g., GBD incidence/prevalence tables for epilepsy)

## Model Specification Files

Outcome model coefficients are stored in JSON files:
- Location: `data/*CohortModelSpec.json`
- Examples: `data/StrokeMIPartitionModel.json`, `data/DementiaModelSpec.json`
- Loaded via: `microsim.data_loader.load_model_spec()`

Structure:
```json
{
  "coefficients": {
    "Intercept": -2.3109,
    "age": 0.0234,
    "sbp": 0.0145,
    ...
  },
  "sigma": 0.5,
  ...
}
```

## Architecture: Repository Pattern

The outcomes module uses a three-tier architecture:

1. **OutcomeModelRepository** (`outcome_model_repository.py`):
   - Central registry mapping OutcomeType → specific ModelRepository
   - Ensures all OutcomeTypes have a registered repository
   - Initialized with parameters like `wmhSpecific=True`

2. **Specific Model Repositories** (e.g., `StrokePartitionModelRepository`):
   - Implements `select_outcome_model_for_person(person)`: Returns appropriate model for person
   - Often gender-specific or based on other person characteristics
   - Example: StrokePartitionModelRepository selects FemaleStrokeModel vs MaleStrokeModel

3. **Outcome Models** (e.g., `StrokePartitionModel`):
   - Implements `get_next_outcome(person)`: Returns Outcome instance or None
   - Implements `generate_next_outcome(person)`: Creates Outcome with phenotype details
   - Uses person's risk factors, random effects, and RNG for predictions

## Integration with Person Class

Outcomes are tracked in `Person._outcomes` dictionary:
```python
person._outcomes = {
    OutcomeType.STROKE: [
        (age_at_outcome, StrokeOutcome_instance),
        (age_at_outcome_2, StrokeOutcome_instance_2),
        ...
    ],
    OutcomeType.MI: [...],
    ...
}
```

Access patterns:
- `person.has_outcome_at_current_age(OutcomeType.STROKE)`: Check if outcome occurred this wave
- `person.has_outcome_during_simulation(OutcomeType.STROKE)`: Check if ever occurred
- `person._stroke`, `person._mi`, etc.: Boolean properties for quick access

Wave semantics:
- Outcomes stored with age at occurrence
- Wave indexing: -1 before first advance, then 0, 1, 2, ...
- Outcome arrays indexed by wave number

## priorToSim outcomes carry age=None

Any outcome flagged `priorToSim=True` is stored with `age=None` instead of a real age:

```python
person._outcomes[OutcomeType.STROKE] = [
    (None, StrokeOutcome(..., priorToSim=True)),   # priorToSim entry — age slot is None
    (65, StrokeOutcome(..., priorToSim=False)),    # in-sim entry — real age
]
```

`Person.add_outcome` enforces this automatically: it stores `None` whenever `outcome.priorToSim` is True.

**Why None instead of a real age?** Fail-loud convention. PriorToSim outcomes have no meaningful "age at event" within the simulation, so any code that incorrectly mixes them into age-based arithmetic crashes with `TypeError` (e.g., `None < int`) rather than silently returning wrong answers.

**How to write code that consumes outcome ages safely:**
- Filter priorToSim entries before doing any age arithmetic. The standard helper is `Person.get_outcomes_during_simulation(outcomeType)`, which returns only in-sim outcomes.
- Equivalently, when iterating, gate on `not outcome.priorToSim` *before* the age comparison (use Python's short-circuiting `and`, not bitwise `&`).
- Functions that legitimately can return `None` (e.g., `get_age_at_last_outcome`) should be checked for `None` by their callers.

**Functions in `Person` that already filter priorToSim correctly:**
`get_outcomes_during_simulation`, `has_outcome_during_simulation`, `has_outcome_during_simulation_prior_to_wave`, `get_outcomes` (with default `inSim=True`), `get_age_at_first_outcome` (default `inSim=True`), `get_age_at_last_outcome_in_sim`, `has_outcome_by_age`, `get_person_years_with_outcome_by_end_of_wave`, `has_incident_event`, `get_ages_with_outcome`.

**Regression coverage:** `test/test_outcome_age_handling.py` pins down the convention across these functions.

## Common Tasks

### Creating a New Outcome Model

From `outcome.py` lines 3-13, the complete workflow is:

1. **Create OutcomeType enum value** in `outcome.py`:
   ```python
   class OutcomeType(Enum):
       ...
       NEW_OUTCOME = "new_outcome"
       _order_ = [..., "NEW_OUTCOME", ...]  # Maintain dependency order!
   ```

2. **(Optional) Create Outcome subclass** if you need to store phenotype data:
   ```python
   # outcomes/new_outcome_outcome.py
   from microsim.outcomes.outcome import Outcome, OutcomeType

   class NewOutcome(Outcome):
       def __init__(self, fatal, specific_phenotype_field):
           super().__init__(OutcomeType.NEW_OUTCOME, fatal)
           self.specific_phenotype_field = specific_phenotype_field
   ```

3. **Create Model class(es)** in `outcomes/new_outcome_model.py`:
   ```python
   from microsim.regression_models.linear_risk_factor_model import LinearRiskFactorModel
   from microsim.common.data_loader import load_model_spec

   class NewOutcomeModel(LinearRiskFactorModel):
       def __init__(self):
           model_spec = load_model_spec("NewOutcomeModelSpec")
           super().__init__(RegressionModel(**model_spec))

       def get_next_outcome(self, person):
           # Return Outcome instance or None
           probability = self.estimate_next_risk(person)
           if person._rng.uniform() < probability:
               return self.generate_next_outcome(person)
           return None

       def generate_next_outcome(self, person):
           # Create and return Outcome
           fatal = self.determine_fatality(person)
           return Outcome(OutcomeType.NEW_OUTCOME, fatal)
   ```

4. **Create ModelRepository** in `outcomes/new_outcome_model_repository.py`:
   ```python
   from microsim.outcomes.new_outcome_model import NewOutcomeModel

   class NewOutcomeModelRepository:
       def __init__(self):
           self._male_model = NewOutcomeModel()
           self._female_model = NewOutcomeModel()

       def select_outcome_model_for_person(self, person):
           return self._female_model if person._gender == "female" else self._male_model
   ```

5. **Register in OutcomeModelRepository** (`outcome_model_repository.py`):
   ```python
   from microsim.outcomes.new_outcome_model_repository import NewOutcomeModelRepository

   class OutcomeModelRepository:
       def __init__(self, wmhSpecific=True, riskScaling=None):
           self._repository = {
               ...
               OutcomeType.NEW_OUTCOME: NewOutcomeModelRepository(),
               ...
           }
   ```

6. **Update Person class** (`person.py`) to track the new outcome:
   - Add property: `@property def _new_outcome(self): ...`
   - Initialize in `_outcomes` dictionary
   - Add any necessary helper methods

### Modifying Existing Outcome Models

**To update model coefficients:**
1. Edit the appropriate JSON spec file in `data/` directory
2. Reload the model (or restart simulation)

**To modify prediction logic:**
1. Edit the model class in `outcomes/<outcome>_model.py`
2. Update `get_next_outcome()` or `generate_next_outcome()` methods
3. Run tests to verify: `poetry run test`

**To add new risk factors to a model:**
1. Update model specification JSON with new coefficient
2. Update model's `estimate_next_risk()` arguments if needed
3. May require changes to `model_argument_transform.py`

### Accessing Outcome Data

**From Person instances:**
```python
# Check if person has had stroke
if person.has_outcome_during_simulation(OutcomeType.STROKE):
    stroke_events = person._outcomes[OutcomeType.STROKE]
    for age, stroke_outcome in stroke_events:
        print(f"Stroke at age {age}, NIHSS: {stroke_outcome.nihss}")

# Check current wave
if person.has_outcome_at_current_age(OutcomeType.MI):
    print("MI occurred this wave")
```

**Population-level aggregation:**
```python
# Count outcomes across population
stroke_count = sum(1 for p in population._people
                   if p.has_outcome_during_simulation(OutcomeType.STROKE))

# Get outcome incidence by year
outcomes_by_wave = population.get_outcome_incidence_by_wave(OutcomeType.STROKE)
```

**Trial outcome assessment:**
See `microsim/trials/claude.md` for detailed trial analysis patterns.

## Testing Patterns

Test files for outcomes follow standard unittest conventions:
- Test individual outcome models for correct probability calculations
- Test repository selection logic (gender-specific, etc.)
- Test outcome generation and phenotype assignment
- Test integration with Person class

**Example test structure:**
```python
import unittest
from microsim.person import Person
from microsim.outcomes.outcome import OutcomeType

class TestStrokeModel(unittest.TestCase):
    def setUp(self):
        self.person = create_test_person()  # Helper function

    def test_stroke_probability(self):
        # Test model produces valid probabilities
        pass

    def test_stroke_outcome_properties(self):
        # Test StrokeOutcome has correct phenotype fields
        pass
```

Helper functions live in `test/helper/` (e.g., `init_vectorized_population_dataframe`).

## Cross-References

- **Main CLAUDE.md**: For overall architecture, Person/Population structure, wave semantics
- **microsim/risk_factors/claude.md**: For risk factor inputs used in outcome prediction
- **microsim/trials/claude.md**: For outcome assessment and statistical analysis in trials
- **person.py**: For outcome storage, tracking, and helper methods
- **model_argument_transform.py**: For converting Person attributes to model inputs
