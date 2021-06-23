from typing import Generic, List, TypeVar
import numpy as np
from microsim.store.base_numpy_data_converter import BaseNumpyDataConverter


S = TypeVar("S")  # _S_tatic data type


class NumpyPersonStore(Generic[S]):
    """Holds Person data in numpy ndarrays."""

    def __init__(self, static_data: List[S], static_data_converter: BaseNumpyDataConverter[S]):
        static_dtype = static_data_converter.get_dtype()
        static_data_arraylike = [static_data_converter.to_row_tuple(d) for d in static_data]
        self._static_data_array = np.array(static_data_arraylike, dtype=static_dtype)
