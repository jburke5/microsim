from dataclasses import dataclass, fields
from typing import Type
import numpy as np
from microsim.person.pytype_to_nptype import pytype_to_nptype
from microsim.store.numpy_data_converter_protocol import NumpyDataConverterProtocol


class DataclassNumpyDataConverter(NumpyDataConverterProtocol):
    """Converts dataclasses into Numpy array rows."""

    def __init__(self, type: Type[dataclass]):
        self._type = type

    def get_dtype(self):
        field_specs = [(f.name, pytype_to_nptype(f.type)) for f in fields(self._type)]
        return np.dtype(field_specs)

    def to_row_tuple(self, obj):
        dtype = self.get_dtype()
        values = tuple(getattr(obj, name) for name in dtype.names)
        return values
