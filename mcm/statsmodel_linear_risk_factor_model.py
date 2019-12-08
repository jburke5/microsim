from functools import reduce
from typing import Callable, Dict, Iterable, List, Tuple
import numpy as np

# TODO: this class needs to be renamed. its no longer interfacing with statsmodel
# conceptually, what it does now is bridge the regression model and the person


def get_argument_transforms(parameter_name: str) -> Tuple[str, List[Callable]]:
    folded_param_name = parameter_name.casefold()
    if folded_param_name.startswith("log"):
        trimmed_param_name = parameter_name[len("log"):]
        prop_name, transforms = get_argument_transforms(trimmed_param_name)
        return (prop_name, transforms + [lambda v: np.log(v)])
    elif folded_param_name.startswith("mean"):
        trimmed_param_name = parameter_name[len("mean"):]
        prop_name, transforms = get_argument_transforms(trimmed_param_name)
        return (prop_name, transforms + [lambda v: np.array(v).mean()])
    elif folded_param_name.startswith("square"):
        trimmed_param_name = parameter_name[len("square"):]
        prop_name, transforms = get_argument_transforms(trimmed_param_name)
        return (prop_name, transforms + [lambda v: v ** 2])
    elif folded_param_name.startswith("base"):
        trimmed_param_name = parameter_name[len("base"):]
        prop_name, transforms = get_argument_transforms(trimmed_param_name)
        return (prop_name, transforms + [lambda v: v[0]])
    elif folded_param_name.startswith("lag"):
        trimmed_param_name = parameter_name[len("lag"):]
        expected_prop_name = trimmed_param_name[0].casefold() + trimmed_param_name[1:]
        return (expected_prop_name, [])  # identity: no transformation
    else:
        expected_prop_name = parameter_name[0].casefold() + parameter_name[1:]
        return (expected_prop_name, [])  # identity: no transformation


def get_all_argument_transforms(parameter_names: Iterable[str]) -> Dict[str, List[Callable]]:
    """Return transform functions for each model parameter, if any"""
    param_transforms = {}
    for param_name in parameter_names:
        prop_name, transforms = get_argument_transforms(param_name)
        if param_name.casefold() != prop_name:
            param_transforms[param_name] = (prop_name, transforms)
    return param_transforms


class StatsModelLinearRiskFactorModel:
    def __init__(self, regression_model, log_transform=False):
        self.standard_errors = regression_model._coefficient_standard_errors
        self.log_transform = log_transform
        if (
            hasattr(regression_model, "_residual_mean")
            and hasattr(regression_model, "_residual_standard_deviation")
        ):
            self.residual_mean = regression_model._residual_mean
            self.residual_standard_deviation = regression_model._residual_standard_deviation

        self.parameters = {**(regression_model._coefficients)}
        self._categoricalParams = {}
        self._nonCategoricalParams = {}
        for param_name, param_val in self.parameters.items():
            if param_name == "Intercept":
                continue
            if "[" in param_name:
                self._categoricalParams[param_name] = param_val
            else:
                self._nonCategoricalParams[param_name] = param_val
        self.argument_transforms = get_all_argument_transforms(self._nonCategoricalParams.keys())

    def draw_from_residual_distribution(self):
        if not hasattr(self, "residual_mean") and hasattr(self, "residual_standard_deviation"):
            raise RuntimeError("Cannot draw from residual distribution: model does not have"
                               " residual information")
        return np.random.normal(
            loc=self.residual_mean,
            scale=self.residual_standard_deviation,
            size=1)[0]

    def strip_categorical_name(self, name):
        stripped_name = "_" + name[:name.index("[")]
        stripped_value = int(name[name.index("[T.") + len("[T."): name.index("]")])
        return (stripped_name, stripped_value)

    def get_intercept(self):
        return self.parameters['Intercept']

    def estimate_next_risk(self, person):
        # TODO: think about what to do with teh hard-coded strings for parameters and prefixes
        linearPredictor = self.get_intercept()

        # for non-categorical parameters this is easy — just add the linear predictor
        for coeff_name, coeff_val in self._nonCategoricalParams.items():
            if coeff_name not in self.argument_transforms:
                model_argument = getattr(person, f"_{coeff_name}")
            else:
                prop_name, transforms = self.argument_transforms[coeff_name]
                prop_value = getattr(person, f"_{prop_name}")
                model_argument = reduce(lambda v, f: f(v), transforms, prop_value)
            if isinstance(model_argument, list) or isinstance(model_argument, np.ndarray):
                model_argument = model_argument[-1]
            linearPredictor += coeff_val * model_argument

        # for categorical params, pick which parameter to add...
        for coeff_name, coeff_val in self._categoricalParams.items():
            stripped_name, matched_categorical_value = self.strip_categorical_name(coeff_name)
            if (matched_categorical_value == getattr(person, stripped_name)):
                linearPredictor += coeff_val

        if (self.log_transform):
            linearPredictor = np.exp(linearPredictor)

        return linearPredictor
