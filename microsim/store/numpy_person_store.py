import numpy as np
from microsim.store.numpy_person_record_proxy import new_person_record_proxy_class
from microsim.store.numpy_population_proxy import NumpyPopulationProxy


def assert_positive_int(value, name):
    if not isinstance(value, int):
        raise TypeError(
            f"Expected `{name}` to be an `int`: received: {value}" f" (type: {type(value)})"
        )
    if value <= 0:
        raise ValueError(f"Expected `{name}` to be a positive integer; received: {value}")


class NumpyPersonStore:
    """Holds Person data in numpy ndarrays."""

    def __init__(
        self,
        num_persons,
        num_ticks,
        static_mapping,
        dynamic_mapping,
        event_mapping,
        iter_person_records,
    ):
        assert_positive_int(num_persons, "num_persons")
        self._num_persons = num_persons

        assert_positive_int(num_ticks, "num_ticks")
        self._num_ticks = num_ticks

        static_shape = (self._num_persons,)
        static_dtype = static_mapping.dtype
        self._static_data_array = np.zeros(static_shape, static_dtype)

        dynamic_shape = (self._num_persons, self._num_ticks + 1)  # + 1 for initial data + k ticks
        dynamic_dtype = dynamic_mapping.dtype
        self._dynamic_data_array = np.zeros(dynamic_shape, dynamic_dtype)

        event_shape = (self._num_persons, self._num_ticks + 1)  # + 1 for initial data + k ticks
        event_dtype = event_mapping.dtype
        self._event_data_array = np.zeros(event_shape, event_dtype)

        self._person_record_proxy_class = new_person_record_proxy_class(
            static_mapping.property_mappings,
            dynamic_mapping.property_mappings,
            event_mapping.property_mappings,
        )

        all_person_record_property_names = (
            static_mapping.property_mappings.keys()
            | dynamic_mapping.property_mappings.keys()
            | event_mapping.property_mappings.keys()
        )
        initial_pop = self.get_population_at(-1, active_indices=np.arange(0, self._num_persons))
        for record, person in zip(iter_person_records(), initial_pop):
            for prop_name in all_person_record_property_names:
                value = getattr(record, prop_name)
                setattr(person.next, prop_name, value)

    def get_num_persons(self):
        """Returns the number of people held in this store."""
        return self._num_persons

    def get_num_ticks(self):
        """Returns the number of ticks for which store can hold data."""
        return self._num_ticks

    def get_population_at(self, t, condition=None, active_indices=None):
        """Returns all records at time `t` that satisfy `condition`."""
        population_proxy = NumpyPopulationProxy(
            self,
            t,
            active_condition=condition,
            active_indices=active_indices,
        )
        return population_proxy

    def get_person_record(self, i, t):
        """Returns the record for person `i` at time `t`."""
        static_row = self._static_data_array[i]
        dynamic_row = self._dynamic_data_array[i, t]
        event_row = self._event_data_array[i, t]
        person_record_proxy = self._person_record_proxy_class(static_row, dynamic_row, event_row)
        return person_record_proxy
