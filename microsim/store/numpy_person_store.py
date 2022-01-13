from types import MappingProxyType
import numpy as np
from microsim.store.numpy_person_proxy import NumpyPersonProxy
from microsim.store.numpy_person_record_proxy import PersonRecordProxyMetaclass
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
        initial_person_records=None,
        *,
        npz_file=None,
        person_record_proxy_class=None,
        person_proxy_class=None,
    ):
        assert_positive_int(num_persons, "num_persons")
        self._num_persons = num_persons

        assert_positive_int(num_ticks, "num_ticks")
        self._num_ticks = num_ticks

        if initial_person_records is not None and npz_file is not None:
            raise ValueError("Can load either initial records or an `.npz` file, but not both")

        if person_record_proxy_class is not None:
            self._person_record_proxy_class = person_record_proxy_class
        else:
            field_metadata = MappingProxyType(
                {
                    "static": static_mapping.property_mappings,
                    "dynamic": dynamic_mapping.property_mappings,
                    "event": event_mapping.property_mappings,
                }
            )
            self._person_record_proxy_class = PersonRecordProxyMetaclass(
                "NumpyPersonRecordProxy", tuple(), {}, field_metadata=field_metadata
            )

        self._person_proxy_class = (
            person_proxy_class if person_proxy_class is not None else NumpyPersonProxy
        )

        static_shape = (self._num_persons,)
        static_dtype = static_mapping.dtype
        dynamic_shape = (self._num_persons, self._num_ticks + 1)  # + 1 for initial data + k ticks
        dynamic_dtype = dynamic_mapping.dtype
        event_shape = (self._num_persons, self._num_ticks + 1)  # + 1 for initial data + k ticks
        event_dtype = event_mapping.dtype

        if npz_file is not None:
            static_array, dynamic_array, event_array = self._load_arrays_from_file(
                npz_file,
                static_shape,
                static_dtype,
                dynamic_shape,
                dynamic_dtype,
                event_shape,
                event_dtype,
            )
            self._static_data_array = static_array
            self._dynamic_data_array = dynamic_array
            self._event_data_array = event_array
        else:
            self._static_data_array = np.zeros(static_shape, static_dtype)
            self._dynamic_data_array = np.zeros(dynamic_shape, dynamic_dtype)
            self._event_data_array = np.zeros(event_shape, event_dtype)

        self._scratch_dynamic_data_array = np.zeros((self._num_persons,), dynamic_dtype)
        self._scratch_event_data_array = np.zeros((self._num_persons,), event_dtype)

        if initial_person_records is not None:
            self._load_initial_records(
                static_mapping, dynamic_mapping, event_mapping, initial_person_records
            )

    def _load_initial_records(
        self, static_mapping, dynamic_mapping, event_mapping, initial_person_records
    ):
        all_person_record_property_names = (
            static_mapping.property_mappings.keys()
            | dynamic_mapping.property_mappings.keys()
            | event_mapping.property_mappings.keys()
        )
        initial_pop = self.get_population_at(t=-1)
        for record, person in zip(initial_person_records, initial_pop):
            for prop_name in all_person_record_property_names:
                value = getattr(record, prop_name)
                setattr(person.next, prop_name, value)

    def _load_arrays_from_file(
        self,
        file,
        static_shape,
        static_dtype,
        dynamic_shape,
        dynamic_dtype,
        event_shape,
        event_dtype,
    ):
        with np.load(file, allow_pickle=False) as data:
            missing_arrays = {"static", "dynamic", "event"} - set(data.keys())
            if missing_arrays:
                raise ValueError(f"Required arrays not found: {sorted(missing_arrays)}")
            static_array = data["static"]
            dynamic_array = data["dynamic"]
            event_array = data["event"]

        self._validate_external_array("static", static_array, static_dtype, static_shape)
        self._validate_external_array("dynamic", dynamic_array, dynamic_dtype, dynamic_shape)
        self._validate_external_array("event", event_array, event_dtype, event_shape)

        return (static_array, dynamic_array, event_array)

    def _validate_external_array(self, array_name, array, dtype, shape):
        cap_array_name = array_name.capitalize()
        if array.dtype != dtype and array.shape != shape:
            raise ValueError(
                f"{cap_array_name} array does not have the required dtype"
                f" ({array.dtype} != {dtype}) nor the required shape"
                f" ({array.shape} != {shape})"
            )
        elif array.dtype != dtype:
            raise ValueError(
                f"{cap_array_name} array does not have the required dtype"
                f" ({array.dtype} != {dtype})"
            )
        elif array.shape != shape:
            raise ValueError(
                f"{cap_array_name} array does not have the required shape"
                f" ({array.shape} != {shape})"
            )

    def get_num_persons(self):
        """Returns the number of people held in this store."""
        return self._num_persons

    def get_num_ticks(self):
        """Returns the number of ticks for which store can hold data."""
        return self._num_ticks

    def get_population_at(self, t):
        """Returns a population containing all person data at time `t`."""
        return NumpyPopulationProxy(self, t)

    def get_scratch_person_record(self, i):
        static_row = self._static_data_array[i]
        dynamic_row = self._scratch_dynamic_data_array[i]
        event_row = self._scratch_event_data_array[i]
        person_record_proxy = self._person_record_proxy_class(static_row, dynamic_row, event_row)
        return person_record_proxy

    def get_person_record(self, i, t):
        """Returns the record for person `i` at time `t`."""
        static_row = self._static_data_array[i]
        dynamic_row = self._dynamic_data_array[i, t]
        event_row = self._event_data_array[i, t]
        person_record_proxy = self._person_record_proxy_class(static_row, dynamic_row, event_row)
        return person_record_proxy

    def get_person_proxy_at(self, i, t, *, scratch_next=False):
        if scratch_next:
            next_record = self.get_scratch_person_record(i)
        else:
            next_record = self.get_person_record(i, t + 1)

        if t == -1:
            cur_prev_records = []
        else:
            cur_prev_records = [self.get_person_record(i, t) for t in range(t + 1)]
        person_proxy = self._person_proxy_class(next_record, cur_prev_records)
        return person_proxy

    def save_to_file(self, file):
        arrays = {
            "static": self._static_data_array,
            "dynamic": self._dynamic_data_array,
            "event": self._event_data_array,
        }
        np.savez_compressed(file, **arrays)
