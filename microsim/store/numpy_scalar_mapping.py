from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class NumpyScalarMapping:
    field_name: str
    field_type: Any
    to_np: Callable[[Any], Any]
    from_np: Callable[[Any], Any]
