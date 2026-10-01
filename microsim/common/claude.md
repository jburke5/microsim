# Common Utilities Documentation

Shared enumerations, data-access helpers, and lightweight value types used throughout the MICROSIM codebase.

## Overview

The `common/` package provides two categories of shared building blocks:

1. **Value types / enumerations** — `AgeScope`, `VariableType`, and `PopulationType`. These are dependency-free and are re-exported from the package `__init__.py` so callers can import them with a single `from microsim.common import ...` statement.

2. **Data-access helpers** — `data_loader.py` functions for resolving file paths and loading model specification JSON files. These are intentionally *not* re-exported from `__init__.py` (they carry a `RegressionModel` dependency); callers import them directly from `microsim.common.data_loader`.

## Directory Structure

- `__init__.py`: Re-exports `AgeScope`, `VariableType`, and `PopulationType` as the public API; explicitly excludes `data_loader` to keep the value-type import dependency-free.
- `age_scope.py`: Frozen dataclass `AgeScope` representing an inclusive age range used for prevalence/incidence pooling and calibration.
- `data_loader.py`: File-path resolution and model specification loading helpers (`get_absolute_datafile_path`, `load_datafile`, `load_model_spec`, `load_regression_model`).
- `population_type.py`: `PopulationType` enum identifying the data source for a population (NHANES, Kaiser, State).
- `variable_type.py`: `VariableType` enum classifying risk factor variables as continuous or categorical.

## Enumerations and Classes

### `AgeScope`

A frozen dataclass (`dataclasses.dataclass(frozen=True)`) representing an inclusive age range.

```
AgeScope()            # all ages (both bounds None)
AgeScope(lo=65)       # age >= 65
AgeScope(lo=70, hi=74)  # 70..74 inclusive
AgeScope(lo=75, hi=75)  # exactly age 75
```

Fields:
- `lo: Optional[int]` — lower bound (inclusive); `None` means unbounded below.
- `hi: Optional[int]` — upper bound (inclusive); `None` means unbounded above.

Key methods/properties:
- `contains(age: int) -> bool` — returns `True` if `age` falls within the scope.
- `label -> str` — a human-readable string (e.g. `"pooled_65_plus"`, `"age_group_70-74"`, `"pooled_overall"`) suitable for use as a dictionary key or report heading.

Construction raises `ValueError` if `lo > hi`.

**Used by:** `population.py` (calibration loops), `population_factory.py` (factory helpers), `outcome_prevalence_model_repository.py` (prevalence target lookups), and calibration tests.

### `PopulationType`

```python
class PopulationType(Enum):
    NHANES = "nhanes"
    KAISER = "kaiser"
    STATE  = "state"
```

Identifies the data source for a population. Used by `PersonFactory`, `PopulationFactory`, `TrialDescription`, and the top-level `microsim` package `__init__.py` to dispatch initialization and loading logic to the correct data pipeline.

### `VariableType`

```python
class VariableType(Enum):
    CONTINUOUS   = "continuous"
    CATEGORICAL  = "categorical"
```

Classifies risk factor variables for population initialization. Used in `PopulationFactory` to partition variable lists when drawing from NHANES or Kaiser data (e.g., `variable_types(varType=VariableType.CATEGORICAL.value, popType=PopulationType.NHANES.value)`).

## Data Loader

All functions in `data_loader.py` resolve paths relative to the `data/` directory at the package root (one level above `common/`).

### `get_absolute_datafile_path(filename) -> str`

Returns the absolute path to a file in `data/`. This is the canonical way to locate data files throughout the codebase.

```python
from microsim.common.data_loader import get_absolute_datafile_path

path = get_absolute_datafile_path("fullyImputedDataset.dta")
# e.g. "/path/to/microsim/data/fullyImputedDataset.dta"
```

**Used by:** `population.py`, `standardized_population.py`, `population_factory.py` (the NHANES `.dta`, the `kaiser/` directory, and the `state/` directory), and any module that opens raw data files directly.

Prefer it over a literal `"microsim/data/..."` string: a relative path only resolves when the process happens to run from the repository root, so it breaks for a notebook in a subdirectory, an installed package, or a scheduled job with its own working directory.

### `load_datafile(filename) -> str`

Opens `data/<filename>` and returns its contents as a string. Used internally by `load_model_spec`.

### `load_model_spec(modelname) -> dict`

Loads `data/<modelname>Spec.json` and returns the parsed JSON as a Python dict. The `modelname` argument is validated against the pattern `^[A-Za-z0-9\-]+$` before constructing the filename; an unsafe name raises `ValueError`.

**Used by:** outcome model classes (`cv_model.py`, `stroke_partition_model.py`, `non_cv_death_model.py`) to load regression coefficients.

### `load_regression_model(modelname) -> RegressionModel`

Calls `load_model_spec` and constructs a `RegressionModel` from the result. Convenience wrapper used by risk factor model repositories (`risk_model_repository.py`, `cohort_risk_model_repository.py`) and `afib_model.py`.

## Importing

Import value types from the package:

```python
from microsim.common import AgeScope, VariableType, PopulationType
```

Import data-loading helpers directly from the module:

```python
from microsim.common.data_loader import (
    get_absolute_datafile_path,
    load_model_spec,
    load_regression_model,
)
```

## Gotchas

- **`data_loader` is not re-exported from `__init__.py`** — importing `from microsim.common import get_absolute_datafile_path` will fail. Always use the concrete module path `microsim.common.data_loader`.
- **Path resolution is relative to the package root**, not the caller's file or the current working directory. This means `get_absolute_datafile_path` works correctly regardless of where Python is invoked.
- **`load_model_spec` expects the bare model name**, not the filename. Pass `"ldl"` to load `data/ldlSpec.json`; passing `"ldlSpec.json"` will fail the safety regex and raise `ValueError`.
- **`AgeScope` is immutable** (`frozen=True`). Create a new instance rather than modifying an existing one.
