# Person Package Documentation

The `person/` package defines the individual agent at the base of the Person → Population → Trial hierarchy, along with factory and filter utilities for constructing and selecting Person instances.

## Overview

`Person` is the fundamental simulation unit in MICROSIM. Each Person instance holds all person-level data — past and present — for a single simulated individual. The class has two responsibilities: predicting a person's future state when external predictive models are supplied, and providing analysis and reporting tools over a person's recorded history. Predictive models are never stored inside the Person; they are passed as arguments to simulation methods.

The hierarchy is:

```
Trial (experimental design)
  ↓ compares
Population (collection of Persons)
  ↓ iterates over
Person (individual agent)
```

## Directory Structure

- `person.py`: The `Person` class — the core agent. Holds all risk-factor, treatment, outcome, and cognition data for one individual. Implements wave-advance logic and all reporting/query methods.
- `person_factory.py`: `PersonFactory` — static methods that build `Person` instances from NHANES or Kaiser row data. Handles column-name mapping, bounds enforcement, post-construction initialization (PVD, AFIB, modality, education), and priorToSim outcome seeding.
- `person_filter.py`: `PersonFilter` — a two-level (dataframe + person-object) filter container. Filters are stored as named callables and applied either before or after Person construction.
- `person_filter_factory.py`: `PersonFilterFactory` — a factory holding a named-filter registry (`filterMap`); builds a `PersonFilter` from a list of filter keys, defaulting to the `adult` (age >= 18) filter.
- `__init__.py`: Re-exports `Person`, `PersonFactory`, `PersonFilter`, and `PersonFilterFactory` so that `from microsim.person import Person` works. All four names are listed in `__all__`. Internal modules must use concrete-path imports (e.g. `from microsim.person.person import Person`) to avoid circular imports.

## Person Structure

### Attribute Groups

`Person.__init__` receives five dictionaries. Their keys are used to set named attributes on the instance; the dictionaries themselves are stored as ordered key lists for iteration:

| Attribute | Type | Description |
|-----------|------|-------------|
| `_staticRiskFactors` | `list[str]` | Ordered list of static risk-factor keys (e.g. `raceEthnicity`, `education`, `gender`, `smokingStatus`, `modality`). The corresponding values are stored as `self._<key>` (scalar). |
| `_dynamicRiskFactors` | `list[str]` | Ordered list of dynamic risk-factor keys (e.g. `age`, `sbp`, `dbp`, `a1c`, etc.). Each value is stored as `self._<key>` — a **list** that grows by one element per wave. |
| `_defaultTreatments` | `list[str]` | Ordered list of default-treatment keys (e.g. `antiHypertensiveCount`, `statin`). Each value is also a **list** that grows per wave, representing usual-care treatments. |
| `_treatmentStrategies` | `dict` | Keyed by `TreatmentStrategiesType.value`. Each value is a sub-dict with at minimum `{"status": None}`. Status tracks `BEGIN / MAINTAIN / END / None` lifecycle for each strategy. |
| `_outcomes` | `dict` | Keyed by `OutcomeType`. Each value is a list of `(age, Outcome)` tuples. Multiple events of the same type are represented by multiple elements. |
| `_randomEffects` | `dict` | Populated by outcome models that require random effects; keys and values are set by those models, not by Person itself. |
| `_rng` | `numpy.random.Generator` | Per-person random number generator initialized from OS entropy. Ensures reproducibility across multiprocessing workers without sharing state. |
| `_name` | scalar | Identifier from the source dataset (e.g. NHANES person ID). Multiple Person instances may share a name. |
| `_index` | int or None | Unique identifier assigned by the containing Population instance; `None` until set externally. |
| `_waveCompleted` | int | Starts at `-1` before any advance. Incremented to `0` after the first complete wave, then `1`, `2`, etc. |

### Wave / Subscript Semantics

A "wave" represents one annual time-step forward. The semantics are explained in the `Person` class docstring (`person.py`, lines 24–41) and the `advance()` docstring (lines 94–99):

- `_waveCompleted` is initialized to `-1` before the first `advance()` call.
- After the first complete advance, `_waveCompleted` becomes `0`; after the second, `1`; and so on.
- List-valued attributes (`_dynamicRiskFactors`, `_defaultTreatments`) use **subscript indices** that match wave numbers: subscript `[0]` is the baseline (pre-simulation) value, subscript `[1]` is the value after wave 1, etc.
- The first call to `advance()` skips `advance_risk_factors` and `advance_treatments` because `_waveCompleted == -1`; it proceeds directly to `advance_treatment_strategies_and_update_risk_factors` then `advance_outcomes`. This is the initialization convention for NHANES-sourced persons.
- Outcome timing: if a person has an event during wave 1, `_outcomes[outcomeType]` contains `(age_at_subscript_1, outcome)`. The age stored with the outcome is `self._current_age` at the moment `add_outcome` is called (i.e. the age at subscript `[-1]`).

### priorToSim Outcomes and age=None

Outcomes that occurred before the simulation started are represented with `outcome.priorToSim == True`. In `add_outcome()` (line 237):

```python
age = None if outcome.priorToSim else self._current_age
self._outcomes[outcome.type].append((age, outcome))
```

This means every priorToSim entry carries `age=None` as a deliberate fail-loud sentinel. Any code that consumes outcome ages must filter out priorToSim entries first (use `has_outcome_prior_to_simulation()`, `get_outcomes_during_simulation()`, or the `inSim` parameter on the query methods).

### `advance()` Signature

```python
person.advance(
    years,                      # int: number of annual waves to advance
    dynamicRiskFactorRepository,
    defaultTreatmentRepository,
    outcomeModelRepository,
    treatmentStrategies=None    # TreatmentStrategyRepository or None
)
```

Each call loops `years` times. Within each iteration (while the person is alive):

1. If `_waveCompleted > -1`: advance dynamic risk factors, then default treatments.
2. Advance treatment strategies and update risk factors in place.
3. Advance outcomes.
4. Increment `_waveCompleted`.

Example:

```python
from microsim.person import PersonFactory
from microsim.risk_factors.initialization_model_repository import InitializationModelRepository

# Build a person from a single NHANES row (x)
person = PersonFactory.get_nhanes_person(x, InitializationModelRepository())

# Advance 5 years with repositories from a Population
person.advance(5, dynamicRiskFactorRepository, defaultTreatmentRepository, outcomeModelRepository)
```

## PersonFactory

`PersonFactory` is a stateless class of `@staticmethod` methods. It never stores state; every method constructs and returns a new `Person`.

### Key Methods

- **`get_person(x, popType, initializationModelRepository, outcomePrevalenceModelRepository)`** — Dispatcher that routes to `get_nhanes_person` or `get_kaiser_person` based on `popType`.

- **`get_nhanes_person(x, initializationModelRepository, outcomePrevalenceModelRepository=None)`** — Builds a Person from a single NHANES dataframe row. Steps:
  1. Calls `get_nhanes_person_init_information(x)` to assemble the five init dicts.
  2. Constructs a `Person`.
  3. Applies `InitializationModelRepository` to set `_pvd`, `_afib` (as single-element lists), and `_modality` (scalar) — these cannot be initialized from the raw NHANES row directly.
  4. If `outcomePrevalenceModelRepository` is provided, calls `person.seed_prevalent_outcomes(...)` to seed priorToSim outcomes via logistic prevalence models.

- **`get_kaiser_person(x, initializationModelRepository=None)`** — Builds a Person from a Kaiser data row. Uses `InitializationModelRepository` (pass a shared instance when constructing many persons; one is built per call when omitted) to fill in `_waist`, `_alcoholPerWeek`, and `_education` (missing from the Kaiser raw data). Then adds WMH, epilepsy, and cognition outcomes via module-level shared model instances.

### Column Name Mapping

Two class-level dictionaries map MICROSIM attribute names to source-data column names:
- `microsimToNhanes`: maps `DynamicRiskFactorsType.SBP.value → "meanSBP"`, etc.
- `microsimToKaiser`: maps Kaiser-specific column names (e.g. `"H1A1c"`, `"TotCholesterol"`).

These allow `PersonFactory` to read heterogeneous source data without requiring callers to rename columns.

## PersonFilter

`PersonFilter` holds two named collections of filter functions:

```
filters["df"]     — functions applied to dataframe rows (before Person construction)
filters["person"] — functions applied to Person objects (after Person construction)
```

Each filter is a `(filterName, filterFunction)` pair added via:

```python
pf = PersonFilter()
pf.add_filter("df", "lowSBPLimit", lambda x: x["sbp"] > 126)
pf.add_filter("person", "noMCI", lambda x: not x.has_mci(inSim=False))
```

Filters can also be removed by name with `rm_filter(filterType, filterName)`.

The two-level design exists for efficiency: dataframe-level filters avoid constructing Person objects for rows that will be rejected, saving both memory and CPU. Person-level filters are used when the acceptance criterion requires running a predictive model (e.g. CV risk threshold) that can only operate on a fully constructed Person.

## PersonFilterFactory

`PersonFilterFactory` exposes a class-level registry, `filterMap`, that maps a short string key to a `(filterType, filterFunction)` pair, so pre-defined filters can be requested by name without writing lambdas:

| Key | Level | Criterion |
|-----|-------|-----------|
| `adult` | `df` | age >= 18 |
| `lowSBPLimit` | `df` | SBP > 126 |
| `lowDBPLimit` | `df` | DBP > 85 |
| `highAntiHypertensivesLimit` | `df` | antiHypertensiveCount <= 3 |
| `highCVLimit` | `person` | one-year CV risk < 0.00477 |
| `noMCI` | `person` | `not person.has_mci(inSim=False)` |
| `hasEpilepsy` | `person` | `person.has_epilepsy()` |

`get_person_filter(filterNames=["adult"])` builds a `PersonFilter` from any list of these keys (each added under its key as its filter name); an unknown key raises `ValueError`. `filterNames` defaults to `["adult"]`, so `get_person_filter()` with no arguments returns the default adult-only `PersonFilter`; pass `[]` for a `PersonFilter` with no filters.

`PopulationFactory` calls `get_person_filter(["adult"])` as the default NHANES filter when `personFilters is None`. Call sites name `["adult"]` explicitly rather than relying on the default, so that what a call selects is readable without opening this factory; the bare `get_person_filter()` form appears only in the examples above, which exist to document the default itself.

When `personFilters` **is** supplied, `get_nhanes_people` warns that NHANES now includes children. The warning is unconditional on what the supplied filter contains: it fires even for a filter built from `["adult"]`, since the caller who passes a filter is the one responsible for the age range of the population and the warning is what says so. Only `personFilters=None`, which gets the `adult` default, is silent.

## Common Gotchas

1. **Wave indexing starts at -1**: `_waveCompleted` is `-1` before the first `advance()` call. After the first complete wave it becomes `0`. Valid wave indices for queries are `0` through `_waveCompleted`.

2. **Per-person RNG**: Each `Person` owns an independent `_rng = np.random.default_rng()` seeded from OS entropy at construction. This ensures each person gets statistically independent draws even under multiprocessing, where processes share memory but not state.

3. **priorToSim outcomes carry `age=None`**: The `add_outcome()` method (line 237 of `person.py`) stores `age=None` for any outcome where `outcome.priorToSim is True`. All code that consumes outcome ages must call `get_outcomes_during_simulation()` or check `outcome.priorToSim` first to avoid operating on `None` ages.

4. **Static risk factors are scalars; dynamic risk factors and treatments are lists**: After `__init__`, `self._age` is a list `[baseline_age]`; `self._raceEthnicity` is a scalar enum. This asymmetry is intrinsic to the design.

5. **AFIB and PVD cannot be read directly from NHANES rows**: These two dynamic risk factors are initialized to `None` during `get_nhanes_person_init_information` and are set afterward by `InitializationModelRepository` within `get_nhanes_person`. Always use `get_nhanes_person` (not the `_init_information` helper alone) to get a fully initialized NHANES person.

6. **`get_kaiser_person` does not accept an `outcomePrevalenceModelRepository`**: Kaiser persons seed WMH, epilepsy, and cognition outcomes inline during factory construction, not through the generic prevalence seeding mechanism.
