from dataclasses import dataclass
from typing import List
from microsim.store.numpy_scalar_mapping import NumpyScalarMapping


@dataclass(frozen=True)
class NumpyStructMapping:
    field_name: str
    scalar_mappings: List[NumpyScalarMapping]
