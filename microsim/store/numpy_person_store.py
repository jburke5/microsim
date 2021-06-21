from typing import Generic, List, TypeVar
import numpy as np
from microsim.store.numpy_data_converter_protocol import NumpyDataConverterProtocol


S = TypeVar("S")  # _S_tatic data type


class NumpyPersonStore(Generic[S]):
    """Holds Person data in numpy ndarrays."""

    def __init__(self, static_data: List[S], static_data_converter: NumpyDataConverterProtocol[S]):
        self._static_data_array = np.array(
            [static_data_converter.to_row_tuple(s) for s in static_data],
            dtype=static_data_converter.get_dtype(),
        )
