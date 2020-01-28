from abc import ABCMeta, abstractmethod
from copy import copy
from typing import Dict, Iterable, List, Tuple
import re

import numpy as np


categorical_param_name_pattern = r"^(?P<propname>[^\[]+)\[T\.(?P<matchingval>[^\]]+)\]"
categorical_param_name_regex = re.compile(categorical_param_name_pattern)


class AbstractBaseTransform(metaclass=ABCMeta):
    """Interface definition for model argument transforms."""
    @abstractmethod
    def apply(self, value):
        raise NotImplementedError()


class IndicatorTransform(AbstractBaseTransform):
    """
    Transform that returns 1 if value matches; else 0.

    matching_value should be either a primitive or a `copy`-able container of primitives.
    """
    def __init__(self, matching_value):
        self._matching_value = copy(matching_value)

    @property
    def matching_value(self):
        return self._matching_value

    def apply(self, value):
        return 1 if value == self._matching_value else 0

    def __eq__(self, other):
        if issubclass(type(other), IndicatorTransform):
            return self._matching_value == other.matching_value
        return False


class LogTransform(AbstractBaseTransform):
    """Returns the log (one or many) of the given value."""
    def apply(self, value):
        return np.log(value)


class MeanTransform(AbstractBaseTransform):
    """Returns the mean of the given value."""
    def apply(self, value):
        return np.array(value).mean()


class SquareTransform(AbstractBaseTransform):
    """Returns the square (one or many) of the given value."""
    def apply(self, value):
        return value ** 2


class FirstElementTransform(AbstractBaseTransform):
    """Returns the first element of the given value."""
    def apply(self, value):
        return value[0]


Transform = AbstractBaseTransform


def get_argument_transforms(
    parameter_name: str,
    max_num_transforms: int = 10,
) -> Tuple[str, List[Transform]]:
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
            prop_transforms.append(LogTransform())
        elif folded_param_name.startswith("mean"):
            trimmed_param_name = trimmed_param_name[len("mean"):]
            prop_transforms.append(MeanTransform())
        elif folded_param_name.startswith("square"):
            trimmed_param_name = trimmed_param_name[len("square"):]
            prop_transforms.append(SquareTransform())
        elif folded_param_name.startswith("base"):
            trimmed_param_name = trimmed_param_name[len("base"):]
            prop_transforms.append(FirstElementTransform())
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


def get_all_argument_transforms(parameter_names: Iterable[str]) -> Dict[str, List[Transform]]:
    """Return transform functions for each model parameter, if any"""
    param_transforms = {}
    for param_name in parameter_names:
        prop_name, transforms = get_argument_transforms(param_name)
        if param_name.casefold() != prop_name:
            param_transforms[param_name] = (prop_name, transforms)
    return param_transforms
