from enum import IntEnum
import numpy as np


def pytype_to_nptype(pytype):
    """Returns a numpy type specifier the given Python type."""
    if pytype is bool:
        return np.bool_
    elif pytype is int:
        return np.int64
    elif pytype is float:
        return np.float64
    elif issubclass(pytype, IntEnum):
        return np.int64
    else:
        raise NotImplementedError(
            f"Conversion to numpy type not implemented for Python type: {pytype}"
        )
