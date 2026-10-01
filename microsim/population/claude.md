# Population Package Documentation

This document provides detailed guidance for working with the population layer in MICROSIM.

## Overview

The `population/` package implements the mid-level of the MICROSIM hierarchy:

```
Trial (experimental design)
  ↓ compares
Population (collection of Person objects)
  ↓ created by
PopulationFactory (NHANES / Kaiser / state builders)
  ↓ People managed via
Person (individual agent)
```

A `Population` instance is essentially a Pandas Series of `Person` objects plus a
self-consistent set of prediction models (`PopulationModelRepository`). It knows how to
advance its people through time in a default (usual-care) manner; a `Trial` injects
experimental treatment strategies on top of that default.

The **Person-first principle** applies throughout: meaningful per-person logic lives on
`Person`; `Population` maps or filters over `self._people` and delegates to `Person`
methods. `Population` methods should not re-implement logic that already exists on
`Person`.

## Directory Structure

- `__init__.py`: Package init. Re-exports `Population`, `PopulationFactory`,
  `PopulationModelRepository`, `PopulationRepositoryType`, `StandardizedPopulation`, and
  `InitializationRepository` so callers can do
  `from microsim.population import Population, PopulationFactory`. Internal modules use
  concrete-path imports (e.g. `from microsim.population.population import Population`) to
  avoid import cycles.
- `population.py`: The `Population` class. Stores `self._people` (Pandas Series of
  `Person` objects), `self._modelRepository` (dict keyed by `PopulationRepositoryType`
  values), and `self._waveCompleted`. Contains the simulation engine, outcome queries,
  incidence/prevalence calculations, age/sex standardization, treatment-strategy queries,
  person-year dataframe construction, and console reporting helpers.
- `population_factory.py`: `PopulationFactory` — static-method-only class that builds
  `Population` instances from NHANES data, Kaiser data, or state-level projections.
  Also contains `calibrate_prevalence`, a root-finding utility for matching priorToSim
  prevalence targets.
- `population_model_repository.py`: `PopulationRepositoryType` enum and
  `PopulationModelRepository` class. The enum defines the four repository keys;
  `PopulationModelRepository` wraps them in a single `_repository` dict.
- `initialization_repository.py`: `InitializationRepository` — supplies the initializers
  run once per person at construction time (GCP cognition model and QALY assignment).
- `standardized_population.py`: `StandardizedPopulation` — loads US mortality-table data
  and builds age-group distributions and per-capita weights used for age/sex-standardized
  rate calculations.

## Repository + Factory Pattern

`PopulationModelRepository` aggregates four sub-repositories, each accessed via the
`PopulationRepositoryType` enum:

| Enum member | `.value` string | Sub-repository type |
|---|---|---|
| `PopulationRepositoryType.STATIC_RISK_FACTORS` | `"staticRiskFactors"` | `CohortStaticRiskFactorModelRepository` |
| `PopulationRepositoryType.DYNAMIC_RISK_FACTORS` | `"dynamicRiskFactors"` | `CohortDynamicRiskFactorModelRepository` |
| `PopulationRepositoryType.DEFAULT_TREATMENTS` | `"defaultTreatments"` | `DefaultTreatmentModelRepository` |
| `PopulationRepositoryType.OUTCOMES` | `"outcomes"` | `OutcomeModelRepository` |

`Population.__init__` receives a `PopulationModelRepository` and stores its inner dict
as `self._modelRepository`. Every access to a sub-repository inside `Population` and
`PopulationFactory` uses `PopulationRepositoryType.<MEMBER>.value` as the key — never
a raw string.

`PopulationFactory` is a collection of `@staticmethod` methods. The two primary
population-creation methods are described in the next section. It also exposes lower-level
helpers (`get_nhanes_people`, `get_kaiser_people`) that return a Pandas Series of `Person`
objects without wrapping them in a `Population`.

## PopulationFactory Method Signatures

### NHANES population

```python
@staticmethod
def get_nhanes_population(
    n=None,
    year=None,
    personFilters=None,
    nhanesWeights=False,
    distributions=False,
    customWeights=None,
    riskScaling=None,
    prevalenceRiskScaling=None,
    maxDraws=None,
) -> Population:
```

Parameters:
- `n`: number of people to sample. Honored in every sampling mode — with `nhanesWeights`,
  with `customWeights`, and with neither (rows are then drawn uniformly). `n=None` means no
  sampling at all: every person of that year who passes the filters is returned. Both
  `nhanesWeights=True` and `customWeights` require an `n`. When given, `n` must be an integer of at
  least 1: a float is refused rather than rounded, `bool` is refused although it subclasses `int`,
  and `n=0` is refused because an empty population advances and reports a prevalence of 0 for every
  outcome instead of failing.
- `year`: NHANES survey year; must be one of `{1999, 2001, 2003, 2005, 2007, 2009, 2011, 2013, 2015, 2017}`,
  or `None` to use every survey year at once. `year=None` is refused with `nhanesWeights=True`: those
  weights are defined for each year on its own and cannot weigh a 1999 person against a 2017 one. A
  pooled draw that has to be weighted needs `customWeights` built for pooling.
- `personFilters`: a `PersonFilter` instance; defaults to an adults-only (age >= 18) filter
  when `None`.
- `nhanesWeights`: if `True`, sample with NHANES survey weights (`WTINT2YR`), which is what makes a
  sample representative of the US population of that survey year. Must be `True` or `False` (numpy
  bools accepted) and **defaults to `False`** — anything else, `None` included, raises. It is inferred
  from nothing. It used to follow `distributions`, which meant a call saying only `distributions=True`
  came out weighted without the word appearing in it, and a reader could not tell what the population
  represented without knowing the rule. Requires an `n`; refused with `customWeights` and with
  `year=None`.
- `distributions`: if `True`, keep the categorical variables and the age of each NHANES row and
  replace only its continuous variables with a draw. This is the construction the state populations
  use: the draw comes from a multivariate Gaussian fit on gender, race ethnicity, education and a
  5-year age window, pooled over all NHANES years, and is then shifted by the difference between the
  mean of the person's group and the mean of that crude group (see `group_key_frame` for what a group
  is). Grouping on four variables rather than all nine is what leaves enough people per group to fit a
  covariance that is not singular — all nine variables span ~10,800 cells for the ~5,400 adults of a
  single year, and ~96% of those fits come out singular. The redraw happens **per sampled row**, not
  once per NHANES row, so no two people come out alike: see gotcha 15.
  `personFilters` are honored: the df-level ones are applied to each row after it has been redrawn, so
  a filter such as SBP > 126 holds for the drawn values the people actually carry, and whatever they
  reject is drawn again. `customWeights` is refused with `distributions=True`. `nhanesWeights` gets no
  special default here — a `distributions=True` population is unweighted unless the call says
  `nhanesWeights=True`, and it usually should, since it is the sampling of the NHANES rows that decides
  who the population is made of.
- `customWeights`: alternative Pandas Series of sampling weights; mutually exclusive with
  `nhanesWeights`; requires `n`.
- `riskScaling`: optional `dict[OutcomeType, float]` applied to per-outcome risk inside
  `OutcomeModelRepository`.
- `prevalenceRiskScaling`: optional `dict[OutcomeType, float]` applied to per-outcome
  priorToSim risk inside `OutcomePrevalenceModelRepository`.

### Kaiser population

```python
@staticmethod
def get_kaiser_population(
    n=1000,
    personFilters=None,
    wmhSpecific=True,
    riskScaling=None,
) -> Population:
```

Parameters:
- `n`: number of people to draw from Kaiser distributions (default 1000).
- `personFilters`: a `PersonFilter` instance; `None` means no filter.
- `wmhSpecific`: if `True`, uses WMH-specific CV outcome models in the
  `OutcomeModelRepository`.
- `riskScaling`: optional `dict[OutcomeType, float]` applied inside
  `OutcomeModelRepository`.

### State population

```python
@staticmethod
def get_state_population(
    year=2030,
    personFilters=None,
    state="OH",
    samplingRate=0.025,
) -> Population:
```

Builds a representative sample of a state's projected population and advances it with the
NHANES model repository. The projection CSV (`data/state/pop_projection_{state}_{year}.csv`,
currently OH 2030 and OH 2035 exist — `year` selects the filename only) supplies the categorical
variables and a head count per demographic cell; everything the projection does not carry is
filled in from NHANES:

- **Ages** are split out of the 5-year age groups uniformly; group 17 (80+) is split over
  80-89 by a decreasing taper (`10/55` down to `1/55`), so nobody is 90+ at baseline —
  their share of the group's count is spread over 80-89. Ages past 89 raise.
- **Default treatments** (statin × `antiHypertensiveCount`) are assigned per the NHANES
  survey-weighted proportions within each (ageGroup, gender, raceEthnicity). The counts
  enumerated are the class constant `PopulationFactory.antiHypertensiveCounts` (0-7, all
  values NHANES holds), used by both the cross product and the proportions so the two
  cannot drift and no observed count is left out of the denominators.
- **Continuous variables** are drawn from the same NHANES Gaussians the
  `distributions=True` path uses (crude fit + group-mean shift, see gotchas 14-16).

Parameters:
- `samplingRate`: the fraction of the state population simulated. Each cell contributes
  `floor(n*samplingRate + 0.5)` people; cells that round to 0 contribute nobody (pandas
  `explode` turns an empty range into a NaN row, which is dropped — without that, every
  small cell yielded one spurious person, inflating OH 2030 by 8.7% and biased toward rare
  cells).
- `personFilters`: honored at both levels. The df-level filters run after the continuous
  draw, since the drawn values are what they must hold for; person-level filters run after
  construction. There is no redraw of rejected rows: each row is a fixed slice of the state
  population, so a filter shrinks the population to the filtered subpopulation at the same
  sampling rate. Unlike NHANES, `None` means no filter — the whole state population,
  children included, is the point of this builder; pass `["adult"]` for adults.
- The projection's `raceEthnicity` codes must be ones NHANES holds people of (1-5): the
  whole pipeline (treatment proportions and risk-factor distributions) is NHANES-based, so
  an unsupported code (e.g. ASIAN, 6) is refused up front with a message naming the codes,
  rather than failing as a `KeyError` deep in an apply.

### Generic dispatcher

```python
@staticmethod
def get_population(popType: PopulationType, **kwargs) -> Population:
```

Routes to `get_nhanes_population`, `get_kaiser_population`, or `get_state_population`
based on `popType`. Passes `**kwargs` through to the chosen method unchanged.

### Prevalence calibration

```python
@staticmethod
def calibrate_prevalence(
    scaleOutcomeType,
    targetOutcomeType,
    target,
    scope,
    popType,
    peopleArgs,
    baselineRiskScaling=None,
) -> float:
```

Uses Brent's method in log-space to find the `OutcomePrevalenceModelRepository`
`riskScaling` on `scaleOutcomeType` such that the realized priorToSim prevalence of
`targetOutcomeType` (within `scope`) equals `target`. Returns the float scaling to pass
as `prevalenceRiskScaling={scaleOutcomeType: scaling}` in a subsequent call to
`get_nhanes_population`. Only `PopulationType.NHANES` is supported.

## Simulation Engine — Serial vs Parallel

```python
population.advance(years, treatmentStrategies=None, nWorkers=1)
```

- `nWorkers=1` calls `advance_serial`, which maps `Person.advance(...)` over
  `self._people` in a single thread.
- `nWorkers > 1` calls `advance_parallel`, which splits `self._people` into
  `nWorkers` sub-populations, farms them out to a `multiprocessing.Pool`, and
  re-concatenates the results.

Each `Person` has its own `_rng` (NumPy `default_rng` instance). Because sub-populations
are independent copies with independent RNG state, parallel execution is reproducible
provided the initial RNG seeds are fixed.

`_waveCompleted` tracks how many years the population has been advanced.
Wave numbering starts at -1 before the first advance; after advancing 1 year,
`_waveCompleted == 0`.

## Creating and Advancing a Population (Example)

```python
from microsim.population import Population, PopulationFactory
from microsim.outcomes.outcome import OutcomeType

# Build a 500-person NHANES 1999 population sampled with survey weights
pop = PopulationFactory.get_nhanes_population(
    n=500,
    year=1999,
    nhanesWeights=True,
)

# Advance 5 years (single-threaded)
pop.advance(5)

# Advance 5 years using 4 worker processes
pop.advance(5, nWorkers=4)

# Query outcomes
stroke_count = pop.get_outcome_count(OutcomeType.STROKE)
dementia_incidence = pop.get_outcome_incidence(OutcomeType.DEMENTIA)

# Print a baseline summary
pop.print_baseline_summary()
```

## Population Attributes and Properties

- `_people`: Pandas Series of `Person` objects. The full simulation state lives here.
- `_n`: population size at construction.
- `_waveCompleted`: integer; -1 before any advance, then increments by the `years`
  argument on each `advance` call.
- `_modelRepository`: dict keyed by `PopulationRepositoryType.*.value` strings; populated
  from the `PopulationModelRepository` passed to `__init__`.
- `_staticRiskFactors` (property): list of static risk factor names registered in the
  static-risk-factor sub-repository.
- `_dynamicRiskFactors` (property): list of dynamic risk factor names registered in the
  dynamic-risk-factor sub-repository.
- `_defaultTreatments` (property): list of default treatment names registered in the
  default-treatment sub-repository.

## InitializationRepository

`InitializationRepository.get_initializers()` returns a dict of two one-time
per-person initializers that run during `Person` construction:

- `"_gcp"`: baseline Global Cognitive Performance score via `GCPModel`.
- `"_qalys"`: initial QALY assignment via `QALYAssignmentStrategy`.

These are passed as `imr` to `PersonFactory.get_nhanes_person` and
`PersonFactory.get_kaiser_person`.

## StandardizedPopulation

`StandardizedPopulation(year=2016)` loads the US mortality table from
`data/us.1969_2017.19ages.adjusted.txt` and exposes:

- `ageGroups`: `{gender_value: [[age, ...], ...]}` — age-group membership lists by gender.
- `populationPercents`: `{gender_value: [proportion, ...]}` — share of the total standard
  population in each age group.
- `populationWeightedStandard`: Pandas DataFrame with `age`, `gender`, and `popWeight`
  columns; used for age-standardized NHANES sampling via
  `PopulationFactory.get_nhanes_age_standardized_population`.

`Population.calculate_mean_age_sex_standardized_incidence` uses `StandardizedPopulation`
internally; callers rarely need to instantiate it directly.

## Gotchas

1. **Repository access via enum, not raw strings.** Always use
   `PopulationRepositoryType.OUTCOMES.value` (which equals `"outcomes"`) as the dict
   key, never the bare string `"outcomes"`. This keeps accesses consistent and
   refactoring safe.

2. **Parallel execution copies the model repository.** `get_sub_populations` calls
   `get_pop_model_repository_copy()` for each worker. The copy constructor for
   `PopulationModelRepository` takes `(dynamicRiskFactorRepository, defaultTreatmentRepository,
   outcomeRepository, staticRiskFactorRepository)` in that order.

3. **Per-Person RNG for reproducibility.** Each `Person._rng` is independent; reproducible
   results in parallel mode require fixing the initial person-level RNG seeds before
   calling `advance_parallel`.

4. **`_waveCompleted` vs. Person wave numbering.** `Population._waveCompleted` counts total
   years advanced. Each `Person` also has its own `_waveCompleted`; the two can diverge if
   a person dies partway through (the Person's wave counter stops advancing, but the
   Population's does not).

5. **NHANES year validation.** `get_nhanes_population` raises `RuntimeError` for any year
   not in `{1999, 2001, 2003, 2005, 2007, 2009, 2011, 2013, 2015, 2017}`. `year=None` skips the year
   filter and uses every survey year at once, but only unweighted — see the next gotcha.

6. **`nhanesWeights` and `customWeights` are mutually exclusive**, `customWeights` is also refused
   with `distributions=True`, `nhanesWeights` is refused with `year=None`, and `nhanesWeights`,
   `customWeights` and `maxDraws` all require an `n`. Every check runs before any other work, so
   an argument combination that cannot be honored fails in 0.000s rather than after the distributions
   have been fit and every row redrawn. `nhanesWeights` and `distributions` are also type-checked,
   because the checks combine them with `&`, where a non-bool either raises out of the operator or
   slips through silently.
   The `customWeights`-with-`distributions` check is deliberately made *first*, ahead of every
   `nhanesWeights` check, since otherwise that combination would be reported as the
   mutually-exclusive one and give the less useful of the two messages.

7. **`distributions=True` costs almost nothing once the caches are warm.** It partitions every NHANES
   year on gender, race ethnicity, education and a 5-year age window and fits a Gaussian per group.
   The first call in a process costs roughly 19s — about 14s of it reading the `.dta` and about 5s
   fitting — and every later call is dominated by building the `Person` objects, not by the draw:

   | n | `distributions=True` | `distributions=False` |
   |---|---|---|
   | 1,000 | 0.16s | 0.08s |
   | 5,000 | 0.58s | 0.32s |
   | 20,000 | 1.83s | 1.31s |

   The redraw is vectorized over the distribution groups rather than the rows. The old advice to
   prefer `distributions=False` for speed no longer holds; choose between them on what you want the
   population to be, not on cost.

   The Gaussians and the fine group means are computed on a survey-weight bootstrap of the
   NHANES rows (`get_nhanesDf_resampled`, fixed seed, cached), so the fitted statistics
   reflect the same `WTINT2YR` weights the person-row sampling uses.

   The fits pool all NHANES years, so each draw's shift also adds a per-(year, gender,
   ageGroup) correction (`get_year_corrections`, cells coarse on purpose — the fine groups
   hold ~2 people per year) that moves the row to its own year's level; rows without a
   `year` column (the state projections) keep the pooled levels.

   `alcoholPerWeek` (drinks/week, continuous) is age-gated: every NHANES row under 18 has 0 drinks,
   so keys below `ALCOHOL_MIN_KEY_AGE` (18) fit their Gaussian without the alcohol dimension and
   everyone drawn from them gets exactly 0; keys at 18 and above include it (fits verified
   non-singular after education pooling). `continuous_variables_for_key_age` is the single source
   of the per-key column set for both the fit and the draw. The state projection CSVs still carry a
   0-3 alcohol category column, but `get_stateDf` drops it: state persons draw alcohol from the
   NHANES Gaussians like everyone else.

8. **Kaiser population attribute set differs from NHANES.** Kaiser includes `afib` and
   `pvd` as categorical variables that NHANES does not; Kaiser omits `education` and
   `alcoholPerWeek`. Code that iterates over the variables of a population must use the set
   matching its type, via `variable_types(varType, popType)`, which reads
   `nhanes_variable_types` or `kaiser_variable_types`. Note that the parallel *attribute*
   dict exists for NHANES only: `get_pop_attributes` returns `nhanes_pop_attributes` for
   NHANES but reaches for a `kaiser_pop_attributes` that is defined nowhere, so its Kaiser
   branch raises `AttributeError`. Nothing calls `get_pop_attributes` today.

9. **`bring_people_to_target_n` is the draw loop, not just a top-up.** Filters drop a row only after
   it has been drawn — the person-level ones only after a whole `Person` has been built from it — so
   the loop keeps drawing until `n` people have passed. Called with an empty `people` it is the whole
   draw (what `get_nhanes_people` does when `n` is given); called with the people of an earlier draw
   it tops that draw up. Every pass must be handed the same `weights=` used by the first, otherwise
   the people added later come from a different (unweighted) distribution than the rest. Its first
   pass draws exactly the shortfall, having no acceptance rate to size itself with yet; later passes
   scale by the observed rate with a 20% margin. While nothing at all has been accepted there is still
   no rate to size with, and the batch doubles rather than repeating the shortfall, so filters that
   accept nobody spend the budget in ~log2 passes instead of one pass per shortfall.

10. **Over-restrictive filters raise instead of hanging.** `bring_people_to_target_n` stops after
    sampling `maxDraws` rows (default `max(100*n, 500)`) and raises a `RuntimeError`, of one of two
    kinds. If some rows did pass, the budget is what ran out: the message reports the observed
    acceptance rate, and filters this restrictive by design need an explicit larger `maxDraws`. If
    none did, no budget is large enough and the message says only that none of the rows sampled
    passed, since the rate carries nothing but zero and asking for a larger `maxDraws` would mislead.
    Note the rate covers both filter levels once `distributions` is passed, since the df-level
    filters run inside this loop too.

11. **`get_nhanesDf` is cached and hands out copies.** Building the frame — reading the `.dta`
    and converting the columns — takes about 14 seconds, and every population build needs it,
    so it is built once into `PopulationFactory._nhanesDf`. Each call returns `.copy()` of the
    cache, never the cached object, because callers mutate what they get back:
    `get_proportionForDefaultTreatments` adds an `ageGroup` column and recasts `age` to int. Never
    return the cached frame directly.

12. **The `name` column is the frame's own index.** `get_nhanesDf` renames the data file's
    `index` column to `name`, and that value is what `Person._name` holds. The rename previously
    targeted a `level_0` column that the data file does not have, so no `name` column existed at
    all and the `distributions=True` path died with a `KeyError`.

13. **`draw_from_distributions` bounds its redraws**, and only the Kaiser path uses it now. Each
    group gets `maxDraws` draws, default `max(1000*size, 50000)`, and exhausting the budget raises
    a `RuntimeError` reporting the group and its acceptance rate. The budget is generous because a
    group whose covariance matrix is singular draws from an alternative group's distribution while
    keeping its own bounds, which can leave the acceptance rate near zero. The NHANES path avoided
    that problem rather than tuning around it, by grouping on four variables instead of nine (see
    the `distributions` parameter above).

14. **Two groupings of NHANES exist, for two different jobs, and they are not the same width.**
    `get_partitioned_nhanes_people_crude` groups on gender, race ethnicity, education and a 5-year age
    window, and is what the Gaussians are fit on. `group_key_frame` defines the group whose *mean* a
    draw is shifted to: those same four (with age as a 5-year age group) plus whether the person is on
    antihypertensives and whether they are physically active, both as yes/no. The second key is
    deliberately narrow. It used to be all nine categorical variables plus an age group plus the
    survey year, which is 39,964 cells for 59,204 adults — the median cell held ONE person, so 90.6%
    of adults were shifted to a "mean" that was one individual's values. That inflated the spread of
    every drawn variable by 11-33% and produced shifts of up to 35 sd. The current key holds 2,530
    cells with a median of 54 people. What it gives up is any contrast it does not carry: a person on
    a statin is no longer centred on lower ldl, since statin is not in the key. The survey year went
    with it, so a single-year population is centred on levels pooled over 1999-2017 rather than on
    that year's — for 1999 that moves the mean of a variable by up to 0.24 sd and narrows the sbp gap
    between people on and off antihypertensives from 18.1 to 15.1 mmHg. `append_dataframe_with_continuous`
    lost its `matchYear` argument along with it.

15. **With `distributions=True` the redraw happens after the sampling, one draw per person.** It used
    to redraw each NHANES row once and then bootstrap `n` people out of that fixed set, so a row drawn
    twice produced two people with byte-identical continuous variables — and weighted sampling makes
    that common, since the survey weights span a 196-fold range. A weighted draw of 20,000 people from
    NHANES 1999 held only ~4,200 distinct risk-factor profiles, one of them stamped out 23 times; it
    now holds 20,000. The draw, the redraw and both levels of filtering all happen inside
    `bring_people_to_target_n`, which keeps drawing until `n` people have passed, because the df-level
    filters have to be checked against the drawn values and those do not exist until a row has been
    sampled. `distributions=False` is unchanged and still bootstraps NHANES rows as it always did.

16. **The bounds hold for the shifted draw, not the raw one.** A draw is made from the crude group's
    Gaussian and then shifted to the mean of the fine group, so the value that is kept is the shifted
    one and that is the one `draw_within_bounds` checks: the shift is handed to it and applied before
    the bounds are tested, rather than added afterwards. Checking the raw draw instead left 27% of the
    people of NHANES 1999 outside the observed range of their own group and 151 of 5448 with a
    negative trig, ldl, hdl or creatinine. Each row is redrawn on its own, since rows sharing a
    distribution do not share a shift, and a row that cannot meet the bounds within `maxAttempts`
    keeps its last draw clipped into them — the shift of such a row comes from a group of very few
    people (see gotcha 14 and `get_group_means_for_dataframe`), and it is about 0.02% of them. The
    count is printed as a warning.

## Integration with the Core Framework

Populations sit between the `Person` layer and the `Trial` layer:

```
Person.advance(years, dynamicRiskFactorRepo, defaultTreatmentRepo,
               outcomeModelRepo, treatmentStrategies)
  ↑ called by
Population.advance_serial / advance_parallel
  ↑ called by
Trial.run()  (once for the control arm, once for the treated arm)
```

`PopulationFactory` is also called directly from `TrialDescription` subclasses
(`NhanesTrialDescription`, `KaiserTrialDescription`) to build the people that populate
trial arms.
