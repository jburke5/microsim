from dataclasses import fields, is_dataclass
from typing import Any, Callable, Optional, Type
import numpy as np
from microsim.person.pytype_to_nptype import pytype_to_nptype
from microsim.store.base_numpy_data_converter import BaseNumpyDataConverter


class DataclassNumpyDataConverter(BaseNumpyDataConverter):
    """Converts dataclasses into Numpy array rows."""

    def __init__(self, type: Type, *, field_pytype_to_nptype: Optional[Callable[[Type], Any]]):
        if not is_dataclass(type):
            raise TypeError(f"Given argument `type` is not a dataclass: {type}")
        self._type = type
        self._field_pytype_to_nptype = (
            field_pytype_to_nptype if field_pytype_to_nptype is not None else pytype_to_nptype
        )

    def get_dtype(self):
        field_specs = [(f.name, self._field_pytype_to_nptype(f.type)) for f in fields(self._type)]
        return np.dtype(field_specs)

    def to_row_tuple(self, obj):
        if not isinstance(obj, self._type):
            raise TypeError(
                f"Given argument `obj` is not an instance of the configured type ({self._type}):"
                f" {obj}"
            )
        dtype = self.get_dtype()
        values = tuple(getattr(obj, name) for name in dtype.names)
        return values
