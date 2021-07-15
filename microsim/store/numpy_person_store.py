from microsim.store.numpy_person_record_proxy import NumpyPersonRecordProxy
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
        num_years,
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

        self._num_years = int(num_years)

        self._static_data_converter = static_data_converter
        static_dtype = self._static_data_converter.get_dtype()
        static_data_arraylike = [self._static_data_converter.to_row_tuple(s) for s in static_data]
        self._static_data_array = np.rec.array(static_data_arraylike, dtype=static_dtype)

        self._dynamic_data_converter = dynamic_data_converter
        dynamic_dtype = self._dynamic_data_converter.get_dtype()
        dynamic_data_arraylike = [
            self._dynamic_data_converter.to_row_tuple(d) for d in dynamic_data
        ]
        dynamic_shape = (self._num_years, self._num_persons)
        self._dynamic_data_array = np.rec.array(np.zeros(dynamic_shape, dtype=dynamic_dtype))
        self._dynamic_data_array[0] = dynamic_data_arraylike

        self._event_data_converter = event_data_converter
        event_dtype = self._event_data_converter.get_dtype()
        event_data_arraylike = [self._event_data_converter.to_row_tuple(e) for e in event_data]
        event_shape = (self._num_years, self._num_persons)
        self._event_data_array = np.rec.array(np.zeros(event_shape, dtype=event_dtype))
        self._event_data_array[0] = event_data_arraylike

    def get_num_persons(self):
        """Returns the number of people held in this store."""
        return self._num_persons

    def get_person_record_at(self, i, t):
        """Returns the combined record for Person `i` at time `t`."""
        static_row = self._static_data_array[i]
        dynamic_row = self._dynamic_data_array[t, i]
        event_row = self._event_data_array[t, i]
        record_proxy = NumpyPersonRecordProxy(
            static_row,
            dynamic_row,
            event_row,
            self._static_data_converter,
            self._dynamic_data_converter,
            self._event_data_converter,
        )
        return record_proxy
