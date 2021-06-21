from typing import Generic, List, TypeVar
import numpy as np



S = TypeVar("S")  # _S_tatic data type


class NumpyPersonStore(Generic[S]):
    """Holds Person data in numpy ndarrays."""

    def __init__(self, static_data: List[S]):
        static_type = type(static_data[0])
        self._static_data_array = np.array(
            [s.as_numpy_arraylike() for s in static_data],
            dtype=static_type.get_numpy_dtype(),
        )
