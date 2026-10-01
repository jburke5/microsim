# Treatment Strategies Module Documentation

For general architecture and project overview, see `../../CLAUDE.md`.

## Overview

The treatment strategies module implements experimental intervention protocols that can be applied to Person instances during simulations. Unlike default treatments (usual care), treatment strategies represent specific trial protocols or intervention algorithms.

Key characteristics:
- **State-based lifecycle**: Each strategy progresses through status transitions (None → BEGIN → MAINTAIN → END → None)
- **Dual modification interface**: Strategies can modify both treatments and risk factors
- **Person-level tracking**: Each person maintains their own strategy status and meds added
- **Population-level application**: Strategies can be applied to entire populations with participation tracking

Treatment strategies integrate with:
- `Person._treatmentStrategies`: Dictionary tracking strategy status and meds added by type
- `Population`: Population-level participation tracking and reporting
- `Trial`: Treatment strategy comparison and outcome assessment

## Treatment Strategy Types

From `treatment_strategies/treatment_strategies.py`:
```python
class TreatmentStrategiesType(Enum):
    BP = "bp"  # Blood pressure management strategies
    STATIN = "statin"  # Statin therapy strategies
    WMD15 = "wmd15"  # White matter disease - 15% target
    WMD20 = "wmd20"  # White matter disease - 20% target
    WMD25 = "wmd25"  # White matter disease - 25% target
```

### BP Treatment Strategies

Located in `bp_treatment_strategies.py`, includes:

**Goal-Based Strategies:**
- `AddBPTreatmentMedsToGoal120`: Target SBP=120, DBP=65 (configurable)
- `jnc8Treatment`: JNC8 guidelines (target 140/90 or 150/90 based on age/diabetes/CKD)
- `jnc8ForHighRisk`: Risk-stratified JNC8 (CV risk > threshold)
- `jnc8ForHighRiskLowBpTarget`: JNC8 with custom BP target

**SPRINT Trial Variants:**
- `SprintTreatment`: SPRINT protocol (CV risk > 7.5%, target SBP=126, DBP=85)
- `SprintForLowerDbpGoalTreatment`: SPRINT with DBP=65
- `SprintForSbpOnlyTreatment`: SBP-only goal
- `SprintForSbpRiskThreshold`: SBP-only with custom risk threshold

**Medication Count Strategies:**
- `AddNBPMedsTreatmentStrategy(n)`: Add exactly n BP medications
- `AddASingleBPMedTreatmentStrategy`: Add 1 medication
- `NoBPTreatment`: Control group with no treatment

**Key Constants:**
- `SBP_MULTIPLIER = 5.5`: SBP reduction per medication (mmHg)
- `DBP_MULTIPLIER = 3.1`: DBP reduction per medication (mmHg)
- `MAX_BP_MEDS = 4`: Maximum BP medications allowed

### Statin Treatment Strategies

Located in `statin_treatment_strategies.py`:

**StatinTreatmentStrategy:**
- Risk-based statin assignment
- Default threshold: 7.5% 10-year CV risk
- Only adds statin if not already on statin
- Does not directly modify LDL or other risk factors
- CV risk reduction handled by cv_model.py based on statinsAdded variable

```python
# Example usage
statin_strategy = StatinTreatmentStrategy(cvRiskCutoff=0.075)
```

### WMD Treatment Strategies

Located in `wmd_treatment_strategies.py`:

Three parallel strategies for white matter disease trials:
- `Wmd15TreatmentStrategy`: Sets wmd15MedsAdded=1
- `Wmd20TreatmentStrategy`: Sets wmd20MedsAdded=1
- `Wmd25TreatmentStrategy`: Sets wmd25MedsAdded=1

All follow same pattern:
- BEGIN: Set wmdXXMedsAdded=1
- END: Delete wmdXXMedsAdded
- Do not directly modify risk factors

## Key Files in This Module

### Core Infrastructure
- `treatment_strategies.py`: Base classes, enums (TreatmentStrategiesType, TreatmentStrategyStatus, CategoricalTreatmentStrategiesType, ContinuousTreatmentStrategiesType)
- `treatment_strategy_repository.py`: Repository pattern (maps strategy type → strategy instance); also exposes `from_string()` factory

### Strategy Implementations
- `bp_treatment_strategies.py`: 11+ BP treatment protocols
- `statin_treatment_strategies.py`: Risk-based statin strategy
- `wmd_treatment_strategies.py`: Three WMD strategies (15%, 20%, 25% targets)

### Related Files
- `../default_treatments/default_treatments.py`: Default treatment enums (STATIN, ANTI_HYPERTENSIVE_COUNT)

## Architecture: Strategy Pattern

The treatment strategies module uses the **Strategy Pattern** with a state-based lifecycle:

### 1. Repository Pattern
`TreatmentStrategyRepository`:
- Dictionary mapping TreatmentStrategiesType values → strategy instances
- Initialized with None values for each strategy type
- Populated at runtime with specific strategy instances
- `from_string(name)` class method: convenience factory for common BP strategies; recognized names: `"1bpMedsAdded"`, `"2bpMedsAdded"`, `"3bpMedsAdded"`, `"4bpMedsAdded"`, `"toGoal120"`, `"sprint"`, `"noTreatment"`

### 2. Strategy Interface

All strategies implement two core methods:

```python
def get_updated_treatments(self, person):
    """Return dictionary of treatment updates to apply"""
    return {}  # e.g., {"bpMedsAdded": 2}


def get_updated_risk_factors(self, person):
    """Return dictionary of risk factor updates to apply"""
    return {}  # e.g., {"sbp": 120, "dbp": 65}
```

### 3. Status Lifecycle State Machine

From `TreatmentStrategyStatus` enum (members: `BEGIN`, `MAINTAIN`, `END`) and Python `None`:
- **None** (Python `None`, not an enum member): Not participating in strategy
- **BEGIN**: First wave of participation (initial application)
- **MAINTAIN**: Continuing participation (ongoing application)
- **END**: Final wave of participation (cleanup)

**Status Transitions:**
```
None → BEGIN → MAINTAIN → MAINTAIN → ... → END → None
```

Managed by `person.update_treatment_strategy_status()` (person.py)

## Integration with Person Class

### Storage Structure

Treatment strategies stored in `Person._treatmentStrategies`:
```python
person._treatmentStrategies = {
    "bp": {"status": MAINTAIN, "bpMedsAdded": 2},
    "statin": {"status": BEGIN, "statinMedsAdded": 1},
    "wmd15": {"status": None},
    "wmd20": {"status": None},
    "wmd25": {"status": None},
}
```

### Core Methods (person.py)

**Orchestration:**
- `advance_treatment_strategies_and_update_risk_factors()`: Main entry point
  - Calls update_treatment_strategy_status()
  - Calls update_treatments() for each strategy
  - Calls update_risk_factors() for each strategy

**Status Management:**
- `update_treatment_strategy_status()`: State machine implementation
  - None → BEGIN: When strategy starts
  - BEGIN → MAINTAIN: After first application
  - MAINTAIN → END: When strategy ends
  - END → None: After cleanup wave

**Update Application:**
- `update_treatments()`: Apply get_updated_treatments() results
- `update_risk_factors()`: Apply get_updated_risk_factors() results

**Query Methods:**
- `is_in_treatment_strategy(strategy_type)`: Check if participating (status is not None)
- `is_in_any_treatment_strategy()`: Check if participating in any strategy
- `has_meds_added(strategy_type)`: Check if actively receiving meds (medsAdded > 0)
- `get_meds_added(strategy_type)`: Return the count of meds added for the strategy
- `get_treatment_strategies_with_participation()`: List active strategies
- `get_treatment_strategies_with_meds_added()`: List strategies with meds
- `_antiHypertensiveCountPlusBPMedsAdded()`: Combined BP med count (for max cap)

### Initialization (PersonFactory)

Created as dictionary with keys for all strategy types:
```python
_treatmentStrategies = {
    "bp": {"status": None},
    "statin": {"status": None},
    "wmd15": {"status": None},
    "wmd20": {"status": None},
    "wmd25": {"status": None},
}
```

## Integration with Population Class

### Population-Level Methods (population.py)

**Participation Tracking:**
- `is_in_treatment_strategy(strategy_type)`: Map across all people
- `is_in_any_treatment_strategy()`: Check if any person in any strategy

**Reporting Methods:**
- `print_lastyear_treatment_strategy_distributions()`: Overall participation report
- `print_lastyear_treatment_strategy_distributions_by_risk()`: Risk-stratified report

### Trial Integration

Treatment strategies enable controlled experiments:
- Compare populations with different treatment strategies
- Assess outcomes using TrialOutcomeAssessor
- See `microsim/trials/claude.md` for trial framework details

## Common Tasks

### Creating a New Treatment Strategy

Complete workflow:

1. **Define strategy class** in appropriate file (or create new file):
   ```python
   # treatment_strategies/new_treatment_strategies.py
   from microsim.treatment_strategies.treatment_strategies import TreatmentStrategiesType


   class NewTreatmentStrategy:
       def __init__(self, parameter1, parameter2):
           self.parameter1 = parameter1
           self.parameter2 = parameter2

       def get_updated_treatments(self, person):
           # Return dict of treatment updates
           return {"newMedsAdded": 1}

       def get_updated_risk_factors(self, person):
           # Return dict of risk factor updates
           # Only if strategy directly modifies risk factors
           return {}
   ```

2. **Add to TreatmentStrategiesType enum** (treatment_strategies.py):
   ```python
   class TreatmentStrategiesType(Enum):
       BP = "bp"
       STATIN = "statin"
       WMD15 = "wmd15"
       WMD20 = "wmd20"
       WMD25 = "wmd25"
       NEW_TREATMENT = "new_treatment"  # Add here
   ```

3. **No manual repository change needed**: `TreatmentStrategyRepository.__init__` iterates `TreatmentStrategiesType` automatically, so adding the new member to the enum is sufficient. The new entry will be initialized to `None` at construction time.

4. **Update PersonFactory** to initialize new strategy (person_factory.py):
   - Add key to _treatmentStrategies dict initialization

5. **Use in simulation setup**:
   ```python
   from microsim.treatment_strategies.new_treatment_strategies import NewTreatmentStrategy

   treatment_strategy_repository = TreatmentStrategyRepository()
   treatment_strategy_repository._repository[TreatmentStrategiesType.NEW_TREATMENT.value] = (
       NewTreatmentStrategy()
   )

   # Apply to population
   population.advance(years=10, treatment_strategies=treatment_strategy_repository)
   ```

### Modifying Existing Treatment Strategies

**To update strategy parameters:**
1. Edit the strategy class constructor or constants
2. Update strategy instantiation in simulation code

**To modify strategy logic:**
1. Edit `get_updated_treatments()` or `get_updated_risk_factors()` methods
2. Test changes using test_treatment_strategy.py pattern

**To add new BP strategy variant:**
1. Subclass `BaseTreatmentStrategy` in bp_treatment_strategies.py
2. Override methods as needed
3. Follow existing patterns (JNC8, SPRINT examples)

### Applying Strategies to Simulations

**Population-level application:**
```python
from microsim.population import PopulationFactory
from microsim.treatment_strategies.treatment_strategy_repository import TreatmentStrategyRepository
from microsim.treatment_strategies.bp_treatment_strategies import SprintTreatment

# Create population
population = PopulationFactory.get_nhanes_population(n=1000, year=1999)

# Configure treatment strategies
treatment_strategies = TreatmentStrategyRepository()
treatment_strategies._repository[TreatmentStrategiesType.BP.value] = SprintTreatment()

# Advance with strategies
population.advance(years=10, treatment_strategies=treatment_strategies)

# Check participation
bp_participation = population.is_in_treatment_strategy(TreatmentStrategiesType.BP.value)
print(f"BP strategy participation: {sum(bp_participation)} people")
```

**Trial-level comparison:**
See `microsim/trials/claude.md` for comparing multiple treatment strategies in controlled trials.

### Accessing Treatment Strategy Data

**From Person instances:**
```python
# Check if person in strategy (pass the string value, not the enum member)
if person.is_in_treatment_strategy(TreatmentStrategiesType.BP.value):
    bp_strategy = person._treatmentStrategies["bp"]
    print(f"Status: {bp_strategy['status']}")
    print(f"BP meds added: {bp_strategy.get('bpMedsAdded', 0)}")

# Check active medication addition
if person.has_meds_added(TreatmentStrategiesType.BP.value):
    print("Person actively receiving BP meds from strategy")

# Get all active strategies
active_strategies = person.get_treatment_strategies_with_participation()
```

**From Population instances:**
```python
# Count participation (pass the string value, not the enum member)
participation = population.is_in_treatment_strategy(TreatmentStrategiesType.BP.value)
participation_count = sum(participation)

# Print distributions
population.print_lastyear_treatment_strategy_distributions()
population.print_lastyear_treatment_strategy_distributions_by_risk()
```

## Default Treatments vs Treatment Strategies

**Default Treatments** (defined in `default_treatments/default_treatments.py`):
- Represent usual care / baseline treatment
- Types: STATIN (boolean), ANTI_HYPERTENSIVE_COUNT (count)
- Stored in `person._defaultTreatments`
- Applied automatically based on person characteristics

**Treatment Strategies** (this module):
- Represent experimental interventions
- More complex logic (risk-based, goal-based, trial protocols)
- Stored in `person._treatmentStrategies`
- Applied explicitly via TreatmentStrategyRepository
- Include status lifecycle (BEGIN → MAINTAIN → END)

Both can coexist: default treatments provide baseline, strategies add experimental interventions on top.

## Testing Patterns

Test file: `test/test_treatment_strategy.py`

**Standard test patterns:**
- Test strategy application to Person
- Test risk-based assignment logic
- Test status transitions (None → BEGIN → MAINTAIN → END)
- Test risk factor modifications
- Test treatment modifications
- Test population-level aggregation

**Example test structure:**
```python
import unittest
from microsim.person import Person
from microsim.treatment_strategies.bp_treatment_strategies import SprintTreatment


class TestBPTreatmentStrategy(unittest.TestCase):
    def setUp(self):
        self.person = create_test_person()  # Helper function
        self.strategy = SprintTreatment()

    def test_sprint_risk_threshold(self):
        # Test that strategy applies to high-risk patients
        pass

    def test_bp_reduction(self):
        # Test that BP is reduced by expected amount
        pass

    def test_status_transitions(self):
        # Test BEGIN → MAINTAIN → END lifecycle
        pass
```

## Cross-References

- **Main CLAUDE.md**: For overall architecture, Person/Population structure
- **microsim/risk_factors/claude.md**: For risk factors modified by strategies
- **microsim/outcomes/claude.md**: For outcomes affected by treatment strategies
- **microsim/trials/claude.md**: For treatment strategy comparison in trials
- **person.py**: For treatment strategy storage, status management, and update methods
- **population.py**: For population-level strategy tracking and reporting
