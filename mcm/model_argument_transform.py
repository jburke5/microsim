from copy import copy
import numpy as np


class IndicatorTransform:
    """
    Transform that returns 1 if value matches; else 0.

    matching_value should be either a primitive or a `copy`-able container of primitives.
    """

    def __init__(self, matching_value):
        self._matching_value = copy(matching_value)

    @property
    def matching_value(self):
        return self._matching_value

    def __call__(self, value):
        return 1 if value == self._matching_value else 0

    def __eq__(self, other):
        if issubclass(type(other), IndicatorTransform):
            return self._matching_value == other.matching_value
        return False


def log_transform(value):
    """Returns the log (one or many) of the given value."""
    return np.log(value)


def mean_transform(value):
    """Returns the mean of the given value."""
    return np.array(value).mean()


def square_transform(value):
    """Returns the square (one or many) of the given value."""
    return value ** 2


def base_transform(value):
    """Returns the base (i.e., first) element of the given value."""
    return value[0]
