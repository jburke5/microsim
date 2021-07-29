from enum import IntEnum
from types import MappingProxyType
import numpy as np
from microsim.store.numpy_scalar_mapping import NumpyScalarMapping
from microsim.store.numpy_struct_mapping import NumpyStructMapping
from microsim.util.get_base_annotations import get_base_annotations


def get_enum_member_value(x):
    return x.value


def identity(x):
    return x


def new_outcome_struct_mapping(field_name):
    scalar_mappings = [
        NumpyScalarMapping("type", "U9", identity, identity),
        NumpyScalarMapping("fatal", np.bool8, identity, bool),
    ]
    return NumpyStructMapping(field_name, scalar_mappings)


def get_scalar_mapping(field_name, pytype):
    if pytype is bool:
        return NumpyScalarMapping(field_name, np.bool8, identity, bool)
    elif pytype is int:
        return NumpyScalarMapping(field_name, np.int64, identity, int)
    elif issubclass(pytype, IntEnum):
        return NumpyScalarMapping(field_name, np.int64, get_enum_member_value, pytype)
    elif pytype is float:
        return NumpyScalarMapping(field_name, np.float64, identity, float)

    raise NotImplementedError(f"Scalar mapping not implemented for Python type: {pytype}")


class NumpyEventSubrecordMapping:
    def __init__(self):
        self._property_mappings = {
            p: new_outcome_struct_mapping(p) for p in ["mi", "stroke", "dementia"]
        }
        dtype_fields = [
            (p, [(s.field_name, s.field_name) for s in m.scalar_fields])
            for p, m in self._property_mappings.items()
        ]
        self._dtype = np.dtype(dtype_fields)

    @property
    def property_mappings(self):
        return MappingProxyType(self._property_mappings)

    @property
    def dtype(self):
        return self._dtype


class NumpySubrecordMapping:
    """Specifies how to map a subrecord type to/from a single Numpy array."""

    def __init__(self, protocol, *, scalar_mapping_factory=None, get_record_properties=None):
        if scalar_mapping_factory is None:
            scalar_mapping_factory = get_scalar_mapping
        if get_record_properties is None:
            get_record_properties = get_base_annotations
        self._property_mappings = {}

        dtype_fields = []
        annotations = get_record_properties(protocol)
        for attr_name, attr_type in annotations:
            field_name = attr_name
            mapping = scalar_mapping_factory(field_name, attr_type)
            dtype_fields.append((mapping.field_name, mapping.field_type))
            self._property_mappings[attr_name] = mapping

        self._property_mappings = MappingProxyType(self._property_mappings)
        self._dtype = np.dtype(dtype_fields)

    @property
    def property_mappings(self):
        return self._property_mappings

    @property
    def dtype(self):
        return self._dtype
