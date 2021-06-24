from typing import Generic, List, TypeVar
import numpy as np
from microsim.store.base_numpy_data_converter import BaseNumpyDataConverter


S = TypeVar("S")  # _S_tatic data type
D = TypeVar("D")  # _D_ynamic data type


class NumpyPersonStore(Generic[S, D]):
    """Holds Person data in numpy ndarrays."""

    def __init__(
        self,
        static_data: List[S],
        static_data_converter: BaseNumpyDataConverter[S],
        dynamic_data: List[D],
        dynamic_data_converter: BaseNumpyDataConverter[D],
    ):
        len_static = len(static_data)
        len_dynamic = len(dynamic_data)
        if len_static != len_dynamic:
            raise ValueError(
                "Lengths of `static_data` and `dynamic_data` args do not match:"
                f" {len_static} != {len_dynamic}"
            )
        self._num_persons = len_static

        static_dtype = static_data_converter.get_dtype()
        static_data_arraylike = [static_data_converter.to_row_tuple(d) for d in static_data]
        self._static_data_array = np.array(static_data_arraylike, dtype=static_dtype)

        dynamic_dtype = dynamic_data_converter.get_dtype()
        dynamic_data_arraylike = [dynamic_data_converter.to_row_tuple(d) for d in dynamic_data]
        self._dynamic_data_array = np.array(dynamic_data_arraylike, dtype=dynamic_dtype, ndmin=3)

    def get_num_persons(self):
        """Returns the number of people held in this store."""
        return self._num_persons
