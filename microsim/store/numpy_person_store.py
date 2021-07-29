import numpy as np
from microsim.store.numpy_person_record_proxy import NumpyPersonRecordProxy
from microsim.store.numpy_population_record_proxy import NumpyPopulationRecordProxy


class NumpyPersonStore:
    """Holds Person data in numpy ndarrays."""

    def __init__(
        self,
        static_data,
        dynamic_data,
        event_data,
        static_mapping,
        dynamic_mapping,
        event_mapping,
        num_ticks,
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

        if not isinstance(num_ticks, int):
            raise TypeError(
                f"Expected `num_ticks` to be an `int`: received: {num_ticks}"
                f" (type: {type(num_ticks)})"
            )
        if num_ticks <= 0:
            raise ValueError(
                f"Expected `num_ticks` to be a positive integer; received: {num_ticks}"
            )
        self._num_ticks = num_ticks

        static_shape = (self._num_persons,)
        static_dtype = static_mapping.dtype
        self._static_data_array = np.zeros(static_shape, static_dtype)

        dynamic_shape = (self._num_ticks + 1, self._num_persons)  # + 1 for initial data + k ticks
        dynamic_dtype = dynamic_mapping.dtype
        self._dynamic_data_array = np.zeros(dynamic_shape, dynamic_dtype)

        event_shape = (self._num_ticks + 1, self._num_persons)  # + 1 for initial data + k ticks
        event_dtype = event_mapping.dtype
        self._event_data_array = np.zeros(event_shape, event_dtype)

    def get_num_persons(self):
        """Returns the number of people held in this store."""
        return self._num_persons

    def get_num_ticks(self):
        """Returns the number of ticks for which store can hold data."""
        return self._num_ticks

    def get_population_record_at(self, t, condition=None, active_indices=None):
        """Returns all records at time `t` that satisfy `condition`."""
        static_rows = self._static_data_array
        dynamic_rows = self._dynamic_data_array[t]
        event_rows = self._event_data_array[t]
        population_proxy = NumpyPopulationRecordProxy(
            static_rows,
            dynamic_rows,
            event_rows,
            self._static_data_converter,
            self._dynamic_data_converter,
            self._event_data_converter,
            active_condition=condition,
            active_indices=active_indices,
        )
        return population_proxy

    def get_population_advance_record_window(self, t, condition=None, active_indices=None):
        """
        Returns a tuple of population records at times `t` and `t+1`.

        Intended for `StorePopulation.advance` or other `advance` functions
        that read from the first population record and write to the second one.

        Note that to guarantee alignment of the two population records,
        `condition` and `active_indices` are only used to construct the first
        (i.e., current, or read) population record. The `active_indices` property
        of this record is used to construct the second (next, or write) record via
        the `active_indices` parameter.
        """

        current_record = self.get_population_record_at(t, condition, active_indices)
        next_record = self.get_population_record_at(
            t + 1,
            active_indices=current_record.active_indices,
        )
        return (current_record, next_record)

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
