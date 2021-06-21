from typing import Protocol, Tuple, TypeVar
import numpy as np


T = TypeVar("T")


class NumpyDataConverterProtocol(Protocol[T]):
    """Converts Python objects into rows of Numpy arrays."""

    def get_dtype(self) -> np.dtype:
        """Returns the numpy dtype that will hold the given data in the backing ndarray."""
        pass

    def to_row_tuple(self, obj: T) -> Tuple:
        """Returns `obj` as a tuple that `np.array` can use as a row."""
        pass
