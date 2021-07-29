from enum import IntEnum
from types import MappingProxyType
import numpy as np
from microsim.store.numpy_scalar_mapping import NumpyScalarMapping
from microsim.util.get_base_annotations import get_base_annotations


def get_enum_member_value(x):
    return x.value


def identity(x):
    return x


def outcome_to_np(outcome):
    if outcome is None:
        return ("", False)
    return (outcome.type, outcome.fatal)


def outcome_from_np(row):
    if row.type == "":
        return None


def new_outcome_struct_mapping(field_name):
    outcome_dtype = [("type", "U9"), ("fatal", np.bool8)]
    return NumpyScalarMapping(field_name, outcome_dtype, outcome_to_np, outcome_from_np)


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
            f: new_outcome_struct_mapping(f) for f in ["mi", "stroke", "dementia"]
        }
        dtype_fields = [(m.field_name, m.field_type) for m in self._property_mappings.values()]
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
        for attr_name, attr_type in annotations.items():
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
