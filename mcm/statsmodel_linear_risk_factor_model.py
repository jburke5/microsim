from functools import reduce
from typing import Callable, Dict, Iterable, List, Tuple
import re
import numpy as np
from mcm.model_argument_transform import (
    base_transform,
    IndicatorTransform,
    log_transform,
    mean_transform,
    square_transform,
)

# TODO: this class needs to be renamed. its no longer interfacing with statsmodel
# conceptually, what it does now is bridge the regression model and the person


categorical_param_name_pattern = r"^(?P<propname>[^\[]+)\[T\.(?P<matchingval>[^\]]+)\]"
categorical_param_name_regex = re.compile(categorical_param_name_pattern)


def get_argument_transforms(
    parameter_name: str,
    max_num_transforms: int = 10,
) -> Tuple[str, List[Callable]]:
    trimmed_param_name = parameter_name
    prop_transforms = []
    reached_base_case = False
    for _ in range(max_num_transforms):
        folded_param_name = trimmed_param_name.casefold()
        categorical_match = categorical_param_name_regex.match(parameter_name)
        if categorical_match is not None:
            expected_prop_name = categorical_match['propname']
            matching_categorical_value = int(categorical_match['matchingval'])
            prop_transforms.append(IndicatorTransform(matching_categorical_value))
            reached_base_case = True
            break
        elif folded_param_name.startswith("log"):
            trimmed_param_name = trimmed_param_name[len("log"):]
            prop_transforms.append(log_transform)
        elif folded_param_name.startswith("mean"):
            trimmed_param_name = trimmed_param_name[len("mean"):]
            prop_transforms.append(mean_transform)
        elif folded_param_name.startswith("square"):
            trimmed_param_name = trimmed_param_name[len("square"):]
            prop_transforms.append(square_transform)
        elif folded_param_name.startswith("base"):
            trimmed_param_name = trimmed_param_name[len("base"):]
            prop_transforms.append(base_transform)
        elif folded_param_name.startswith("lag"):
            trimmed_param_name = trimmed_param_name[len("lag"):]
            expected_prop_name = trimmed_param_name[0].casefold() + trimmed_param_name[1:]
            # no transform: identity
            reached_base_case = True
            break
        else:
            expected_prop_name = trimmed_param_name[0].casefold() + trimmed_param_name[1:]
            # no transform: identity
            reached_base_case = True
            break
    if not reached_base_case:
        raise RuntimeError(
            f"Error occurred while parsing parameter name ('{parameter_name}') for model argument"
            f" transforms: exceeded maximum number of iterations ({max_num_transforms}) without"
            " reaching a base case.",
            parameter_name,
            max_num_transforms,
        )
    return (expected_prop_name, reversed(prop_transforms))


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
        self.non_intercept_params = {k: v for k, v in self.parameters.items() if k != 'Intercept'}
        self.argument_transforms = get_all_argument_transforms(self.non_intercept_params.keys())

    def draw_from_residual_distribution(self):
        if not hasattr(self, "residual_mean") and hasattr(self, "residual_standard_deviation"):
            raise RuntimeError("Cannot draw from residual distribution: model does not have"
                               " residual information")
        return np.random.normal(
            loc=self.residual_mean,
            scale=self.residual_standard_deviation,
            size=1)[0]

    def get_intercept(self):
        return self.parameters['Intercept']

    def estimate_next_risk(self, person):
        # TODO: think about what to do with teh hard-coded strings for parameters and prefixes
        linearPredictor = self.get_intercept()

        for coeff_name, coeff_val in self.non_intercept_params.items():
            if coeff_name not in self.argument_transforms:
                model_argument = getattr(person, f"_{coeff_name}")
            else:
                prop_name, transforms = self.argument_transforms[coeff_name]
                prop_value = getattr(person, f"_{prop_name}")
                model_argument = reduce(lambda v, f: f(v), transforms, prop_value)
            if isinstance(model_argument, list) or isinstance(model_argument, np.ndarray):
                model_argument = model_argument[-1]
            linearPredictor += coeff_val * model_argument

        if (self.log_transform):
            linearPredictor = np.exp(linearPredictor)

        return linearPredictor
