import numpy as np


class NumpyPersonStore:
    """Holds Person data in numpy ndarrays."""

    def __init__(
        self,
        num_persons,
        iter_person_records,
        num_years,
        static_data_converter,
        dynamic_data_converter,
        event_data_converter,
    ):
        self._num_persons = num_persons
        self._num_years = num_years

        self._static_data_converter = static_data_converter
        self._dynamic_data_converter = dynamic_data_converter
        self._event_data_converter = event_data_converter

        static_dtype = self._static_data_converter.get_dtype()
        static_shape = (self._num_persons,)
        self._static_data_array = np.empty(static_shape, dtype=static_dtype)

        dynamic_dtype = self._dynamic_data_converter.get_dtype()
        dynamic_shape = (self._num_persons, self._num_years)
        self._dynamic_data_array = np.empty(dynamic_shape, dtype=dynamic_dtype)

        event_dtype = self._event_data_converter.get_dtype()
        event_shape = (self._num_persons, self._num_years)
        self._event_data_array = np.empty(event_shape, dtype=event_dtype)

        for i, person_record in enumerate(iter_person_records):
            self._static_data_array[i] = self._static_data_converter.to_row_tuple(person_record)
            self._dynamic_data_array[i] = self._dynamic_data_converter.to_row_tuple(person_record)
            self._event_data_array[i] = self._event_data_converter.to_row_tuple(person_record)

    def get_num_persons(self):
        """Returns the number of people held in this store."""
        return self._num_persons
