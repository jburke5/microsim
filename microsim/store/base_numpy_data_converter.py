from abc import ABCMeta, abstractmethod
from typing import Generic, Tuple, TypeVar
import numpy as np


T = TypeVar("T")


class BaseNumpyDataConverter(Generic[T], metaclass=ABCMeta):
    """Converts Python objects into rows of Numpy arrays."""

    @abstractmethod
    def get_property_names(self):
        """Python property names this converter maps to Numpy rows."""
        raise NotImplementedError("Abstract method not implemented: get_property_names")

    @abstractmethod
    def get_dtype(self) -> np.dtype:
        """Returns the numpy dtype that will hold the given data in the backing ndarray."""
        raise NotImplementedError("Abstract method not implemented: get_dtype")

    @abstractmethod
    def to_row_tuple(self, obj: T) -> Tuple:
        """Returns `obj` as a tuple that `np.array` can use as a row."""
        raise NotImplementedError("Abstract method not implemented: to_row_tuple")

    @abstractmethod
    def get_value_from_row(self, row, field_name):
        """
        Returns value for given field from a Numpy row.

        Inverse of `to_row_tuple` for each field for which this converter is responsible.
        """
        raise NotImplementedError("Abstract method not implemented: get_value_from_row")
