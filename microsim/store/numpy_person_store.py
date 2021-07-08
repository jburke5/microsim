import numpy as np


class NumpyPersonStore:
    """Holds Person data in numpy ndarrays."""

    def __init__(
        self,
        static_data,
        static_data_converter,
        dynamic_data,
        dynamic_data_converter,
        event_data,
        event_data_converter,
    ):
        len_static = len(static_data)
        len_dynamic = len(dynamic_data)
        len_event = len(event_data)
        if not (len_static == len_dynamic == len_event):
            raise ValueError(
                "Lengths of `static_data`, `dynamic_data`, and `event_data` args do not match:"
                f" {len_static}, {len_dynamic}, {len_event}"
            )
        self._num_persons = len_static  # lengths asserts to be the same: chose static arbitrarily

        static_dtype = static_data_converter.get_dtype()
        static_data_arraylike = [static_data_converter.to_row_tuple(s) for s in static_data]
        self._static_data_array = np.array(static_data_arraylike, dtype=static_dtype)

        dynamic_dtype = dynamic_data_converter.get_dtype()
        dynamic_data_arraylike = [dynamic_data_converter.to_row_tuple(d) for d in dynamic_data]
        self._dynamic_data_array = np.array(dynamic_data_arraylike, dtype=dynamic_dtype, ndmin=3)

        event_dtype = event_data_converter.get_dtype()
        event_data_arraylike = [event_data_converter.to_row_tuple(e) for e in event_data]
        self._event_data_array = np.array(event_data_arraylike, dtype=event_dtype, ndmin=3)

    def get_num_persons(self):
        """Returns the number of people held in this store."""
        return self._num_persons
