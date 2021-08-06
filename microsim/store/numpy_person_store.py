import numpy as np
from microsim.store.numpy_field_proxy import NumpyFieldProxy
from microsim.store.numpy_population_record_proxy import NumpyPopulationRecordProxy


def assert_positive_int(value, name):
    if not isinstance(value, int):
        raise TypeError(
            f"Expected `{name}` to be an `int`: received: {value}" f" (type: {type(value)})"
        )
    if value <= 0:
        raise ValueError(f"Expected `{name}` to be a positive integer; received: {value}")


def assert_unique_prop_names(static_prop_names, dynamic_prop_names, event_prop_names):
    static_dynamic_overlap = static_prop_names & dynamic_prop_names
    static_event_overlap = static_prop_names & event_prop_names
    dynamic_event_overlap = dynamic_prop_names & event_prop_names

    if static_dynamic_overlap | static_event_overlap | dynamic_event_overlap:
        overlap_list = [
            (("static", "dynamic"), static_dynamic_overlap),
            (("static", "event"), static_event_overlap),
            (("dynamic", "event"), dynamic_event_overlap),
        ]
        overlap_dict = {k: v for k, v in overlap_list if v}
        raise ValueError(f"Duplicate property names across subrecords: {overlap_dict}")


def proxy_attrs_from_props(property_mappings, row_attr_name):
    attrs = {}
    for prop_name, mapping in property_mappings.items():
        attrs[prop_name] = NumpyFieldProxy(
            row_attr_name,
            mapping.field_name,
            mapping.to_np,
            mapping.from_np,
        )
    return attrs


def new_person_record_proxy_class(static_props, dynamic_props, event_props):
    assert_unique_prop_names(static_props.keys(), dynamic_props.keys(), event_props.keys())

    static_attrs = proxy_attrs_from_props(static_props, "_static_row")
    dynamic_attrs = proxy_attrs_from_props(dynamic_props, "_dynamic_row")
    event_attrs = proxy_attrs_from_props(event_props, "_event_row")

    def person_record_proxy_init(self, static_row, dynamic_row, event_row):
        self._static_row = static_row
        self._dynamic_row = dynamic_row
        self._event_row = event_row

    base_attrs = {"__init__": person_record_proxy_init}
    proxy_class_attrs = {**static_attrs, **dynamic_attrs, **event_attrs, **base_attrs}
    person_record_proxy_class = type("NumpyPersonRecordProxy", tuple(), proxy_class_attrs)
    return person_record_proxy_class


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

        dynamic_shape = (self._num_ticks + 1, self._num_persons)  # + 1 for initial data + k ticks
        dynamic_dtype = dynamic_mapping.dtype
        self._dynamic_data_array = np.zeros(dynamic_shape, dynamic_dtype)

        event_shape = (self._num_ticks + 1, self._num_persons)  # + 1 for initial data + k ticks
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
        pop_proxy = self.get_population_at(0, active_indices=np.arange(0, self._num_persons))
        for record, dest_proxy in zip(iter_person_records(), pop_proxy):
            for prop_name in all_person_record_property_names:
                value = getattr(record, prop_name)
                setattr(dest_proxy, prop_name, value)

    def get_num_persons(self):
        """Returns the number of people held in this store."""
        return self._num_persons

    def get_num_ticks(self):
        """Returns the number of ticks for which store can hold data."""
        return self._num_ticks

    def get_population_at(self, t, condition=None, active_indices=None):
        """Returns all records at time `t` that satisfy `condition`."""
        static_rows = self._static_data_array
        dynamic_rows = self._dynamic_data_array[t]
        event_rows = self._event_data_array[t]
        population_proxy = NumpyPopulationRecordProxy(
            static_rows,
            dynamic_rows,
            event_rows,
            self._person_record_proxy_class,
            active_condition=condition,
            active_indices=active_indices,
        )
        return population_proxy
