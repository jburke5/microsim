# Regression Models Package Documentation

Reusable statistical model building blocks — coefficient containers and person-to-prediction bridge classes — shared by both `risk_factors/` and `outcomes/`.

## Overview

The `regression_models/` package provides two complementary layers:

1. **Data containers** (`RegressionModel`, `CoxRegressionModel`): passive holders of regression coefficients, standard errors, and residual/hazard parameters loaded from `data/*ModelSpec.json` files.
2. **Bridge classes** (`LinearRiskFactorModel` and its subclasses): wrap a container, read a `Person`'s attributes via `model_argument_transform.py`, and compute the next risk-factor value or outcome probability.

These classes are **generic** — they are not specific to risk factors or outcomes. The same `LinearRiskFactorModel` is instantiated both by `RiskModelRepository` to advance a blood-pressure trajectory and by outcome model repositories to compute stroke probability. For how each domain wires them up, see `../risk_factors/claude.md` and `../outcomes/claude.md`.

## Directory Structure

- `__init__.py`: Re-exports all public classes (`RegressionModel`, `CoxRegressionModel`, `LinearRiskFactorModel`, `LogisticRiskFactorModel`, `RandInterceptLogisticRiskFactorModel`, `CoxRiskFactorModel`, `RelativeRiskFactorModel`, `LinearProbabilityRiskFactorModel`, `RoundedLinearRiskFactorModel`). Import from `microsim.regression_models` rather than individual module paths.
- `regression_model.py`: Base coefficient container (`RegressionModel`).
- `cox_regression_model.py`: Cox-specific coefficient container (`CoxRegressionModel`); subclass of `RegressionModel`.
- `linear_risk_factor_model.py`: Core bridge class (`LinearRiskFactorModel`); all other bridge classes inherit from it.
- `logistic_risk_factor_model.py`: Binary-outcome bridge (`LogisticRiskFactorModel`); applies inverse logit to the linear predictor.
- `rand_intercept_logistic_risk_factor_model.py`: Logistic bridge with a per-person random intercept (`RandInterceptLogisticRiskFactorModel`); used for longitudinal binary outcomes.
- `cox_risk_factor_model.py`: Cox proportional-hazards bridge (`CoxRiskFactorModel`); converts the linear predictor and baseline cumulative hazard into an annual event probability.
- `relative_risk_factor_model.py`: Relative-risk bridge (`RelativeRiskFactorModel`); exponentiates the linear predictor; designed for use in multinomial logistic regression (note: relative risks are not odds ratios).
- `linear_probability_risk_factor_model.py`: Linear-probability bridge (`LinearProbabilityRiskFactorModel`); adds a residual draw then thresholds at 0.5 to return a boolean.
- `rounded_linear_risk_factor_model.py`: Rounded-linear bridge (`RoundedLinearRiskFactorModel`); adds a residual draw then rounds to the nearest non-negative integer; used for count-like risk factors (e.g., medication counts).
- `model_argument_transform.py`: Parameter-name parsing and transform dispatch; converts coefficient key strings (e.g. `"logAge"`, `"meanSbp"`, `"baseLdl"`) into the correct Python operation applied to a `Person` attribute.

## Coefficient Containers

### `RegressionModel`

Passive data container for standard regression models.

```python
RegressionModel(
    coefficients,                  # dict: {str: float}
    coefficient_standard_errors,   # dict: {str: float}
    residual_mean,                 # float
    residual_standard_deviation,   # float
)
```

**Key methods:**
- `to_json()`: Serialize to JSON string (keys: `coefficients`, `coefficient_standard_errors`, `residual_mean`, `residual_standard_deviation`).
- `write_json(filepath)`: Write JSON to file.

Instances are constructed by repositories from the `data/*ModelSpec.json` specification files and passed directly to bridge-class constructors.

### `CoxRegressionModel`

Subclass of `RegressionModel` for Cox proportional-hazards models. Stores the same four fields, but `residual_mean` is interpreted as `_one_year_linear_cumulative_hazard` and `residual_standard_deviation` as `_one_year_quad_cumulative_hazard` (coefficients of the linear and quadratic terms of the baseline cumulative hazard approximation). The JSON serialization reuses the base `residual_mean`/`residual_standard_deviation` keys for compatibility.

## Bridge Classes

All bridge classes inherit from `LinearRiskFactorModel`. The hierarchy is:

```
LinearRiskFactorModel
  ├── CoxRiskFactorModel
  ├── LogisticRiskFactorModel
  │     └── RandInterceptLogisticRiskFactorModel
  ├── RelativeRiskFactorModel
  ├── LinearProbabilityRiskFactorModel
  └── RoundedLinearRiskFactorModel
```

### `LinearRiskFactorModel`

The core bridge. Wraps a `RegressionModel`, uses `model_argument_transform.py` to resolve each coefficient key to a `Person` attribute, computes the dot-product linear predictor, and optionally exponentiates it (`log_transform=True`).

**Constructor:**
```python
LinearRiskFactorModel(regression_model, log_transform=False)
```

**Key methods:**
- `estimate_next_risk(person, rng=None, withResidual=False)`: Returns the linear predictor (optionally exponentiated if `log_transform=True`; optionally adds a residual draw from `N(residual_mean, residual_std)` when `withResidual=True`).
- `get_risk_for_person(person, rng, years)`: Thin wrapper around `estimate_next_risk`; provided for interface consistency with bridge subclasses.
- `draw_from_residual_distribution(rng)`: Draws one sample from `N(residual_mean, residual_standard_deviation)` using the supplied RNG.
- `get_intercept()`: Returns `parameters["Intercept"]`; overridden by `CoxRiskFactorModel` to return `0`.
- `get_manual_parameters()`: Hook for subclasses to inject coefficient-value pairs that are not stored in the JSON spec (returns `{}` by default).
- `get_model_argument_for_coeff_name(coeff_name, person)`: Resolves a single coefficient name to a numeric value from the `Person`, applying any transforms registered in `argument_transforms`.

Interaction terms (coefficient names containing `"#"`) are handled by splitting on `"#"` and multiplying the resolved argument values together.

### `CoxRiskFactorModel`

Wraps a `CoxRegressionModel`. Computes event probability for a one-year interval using the Cox model formula:

```
P(event in year t) = H₀(t-1, t) × exp(linear_predictor)
```

where the baseline cumulative hazard `H₀` is approximated as a quadratic polynomial in years-in-simulation.

**Key methods beyond the base:**
- `linear_predictor(person)`: Returns the raw linear predictor (no intercept).
- `get_cumulative_hazard_for_interval(intervalStart, intervalEnd)`: Baseline cumulative hazard between two time points.
- `get_cumulative_hazard_for_years_in_sim(yearsInSim)`: Convenience wrapper for the interval `[yearsInSim-1, yearsInSim]`.
- `get_risk_for_person(person, years)`: Returns the annual event probability used by outcome model repositories.

### `LogisticRiskFactorModel`

Applies the inverse logit (sigmoid) to the linear predictor to produce a probability in `[0, 1]`. Clamps near 0 for linear predictors below −10 and near 1 above +10 to avoid overflow.

**Key methods:**
- `estimate_linear_predictor(person)`: Returns the raw linear predictor.
- `logit(linearRisk)`: Applies the numerically-stable sigmoid.
- `estimate_next_risk(person)`: Returns `logit(linear_predictor)`.

Used for binary risk factor transitions (e.g., atrial fibrillation onset) and binary outcome probabilities.

### `RandInterceptLogisticRiskFactorModel`

Extends `LogisticRiskFactorModel` with a person-specific random intercept stored in `person._randomEffects[rand_intercept_name]`. The random-intercept standard deviation and mean are read from the `RegressionModel`'s `residual_standard_deviation` and `residual_mean` fields.

**Constructor:**
```python
RandInterceptLogisticRiskFactorModel(
    regression_model,
    log_transform=False,
    rand_intercept_name=None,   # key into person._randomEffects
)
```

**Key methods:**
- `estimate_next_risk(person)`: Adds the person's stored random intercept to the linear predictor before applying the sigmoid.

Used for outcome models (e.g., dementia) where individuals have persistent unobserved heterogeneity.

### `RelativeRiskFactorModel`

Exponentiates the linear predictor to produce a relative risk (not an odds ratio). Designed for use in multinomial logistic regression implementations.

**Key method:**
- `estimate_rel_risk(person)`: Returns `exp(linear_predictor)`.

### `LinearProbabilityRiskFactorModel`

Linear probability model: computes the linear predictor, adds a residual draw from the model's residual distribution (using `person._rng`), then thresholds at 0.5 to return a boolean.

**Key method:**
- `estimate_next_risk(person)`: Returns `True` if `linear_predictor + residual > 0.5`, else `False`.

### `RoundedLinearRiskFactorModel`

Linear model for non-negative integer outcomes: computes the linear predictor, adds a residual draw, rounds to the nearest integer, and clamps to `0` if negative.

**Key method:**
- `estimate_next_risk(person)`: Returns `max(0, round(linear_predictor + residual))`.

Used for count-like risk factors such as antihypertensive medication counts.

## model_argument_transform.py

This module is the bridge between coefficient-name strings (as they appear in the JSON spec) and the actual numeric values extracted from a `Person`. It is called once during `LinearRiskFactorModel.__init__` to pre-compute transform chains for every non-intercept coefficient.

### Transform Classes

All transforms implement `AbstractBaseTransform` with a single `apply(value)` method:

| Class | Effect |
|---|---|
| `IndicatorTransform` | Returns `1` if `value == matching_value`, else `0` |
| `LogTransform` | `np.log(value)` |
| `MeanTransform` | `np.array(value).mean()` |
| `SquareTransform` | `value ** 2` |
| `FirstElementTransform` | `value[0]` (baseline/first wave) |

`Transform` is an alias for `AbstractBaseTransform`.

### Coefficient-Name Parsing

`get_argument_transforms(parameter_name)` parses a coefficient key string into `(prop_name, [transforms])` by stripping recognized prefixes left-to-right:

| Prefix (case-insensitive) | Transform appended |
|---|---|
| `log` | `LogTransform` |
| `mean` | `MeanTransform` |
| `square` | `SquareTransform` |
| `base` | `FirstElementTransform` |
| `lag` | no-op (identity); stops parsing |
| `propname[T.val]` pattern | `IndicatorTransform` with integer `val`; stops parsing |
| anything else | identity; stops parsing |

Transforms are applied in order (outer prefix first, e.g. `"logMeanSbp"` → `MeanTransform` applied to `person._sbp`, then `LogTransform` applied to the result).

### Public API

```python
from microsim.regression_models.model_argument_transform import (
    get_argument_transforms,    # single parameter name → (prop_name, [Transform])
    get_all_argument_transforms,# Iterable[str] → {param_name: (prop_name, [Transform])}
)
```

**`get_argument_transforms(parameter_name)`**
- Returns `(expected_prop_name, transforms)` for one coefficient key.

**`get_all_argument_transforms(parameter_names)`**
- Calls `get_argument_transforms` for each name.
- Omits names where the lowercased form equals the resolved `prop_name` (i.e., names that need no transformation).
- Returns a `dict` mapping coefficient names that require transformation to `(prop_name, [transforms])` tuples.

Example — how `"logMeanSbp"` is resolved:
```python
prop_name, transforms = get_argument_transforms("logMeanSbp")
# prop_name  → "sbp"
# transforms → [MeanTransform(), LogTransform()]
# Applied as: LogTransform().apply(MeanTransform().apply(person._sbp))
```

## Integration with JSON Model Specifications

Model specifications are stored in `data/*CohortModelSpec.json`. Each spec file contains a `coefficients` dict whose keys must match the naming conventions understood by `model_argument_transform.py`. The JSON is loaded by repositories (`CohortRiskModelRepository` in `risk_factors/`, outcome model repositories in `outcomes/`) to construct `RegressionModel` instances, which are then wrapped in the appropriate bridge class.

The mapping is:

```
data/*CohortModelSpec.json
  ↓ loaded by
CohortRiskModelRepository / OutcomeModelRepository
  ↓ constructs
RegressionModel (or CoxRegressionModel)
  ↓ wrapped by
LinearRiskFactorModel (or subclass)
  ↓ called per-person in
RiskModelRepository.advance_person() / OutcomeModelRepository.get_outcome_for_person()
```

For details on how risk-factor repositories wire these models, see `../risk_factors/claude.md`. For outcome model wiring, see `../outcomes/claude.md`.

## Cross-References

- **Main architecture**: `../../CLAUDE.md`
- **Risk factor models**: `../risk_factors/claude.md` — how bridge classes are instantiated and called during risk factor advancement
- **Outcome models**: `../outcomes/claude.md` — how bridge classes are used to compute outcome probabilities
- **Person class**: `../person/person.py` — source of attribute values (`_age`, `_sbp`, `_randomEffects`, `_rng`, etc.) consumed by bridge classes
- **Model spec files**: `../data/*CohortModelSpec.json` — JSON coefficient sources for both NHANES and Kaiser cohorts
