from types import MappingProxyType
from microsim.store.numpy_field_proxy import NumpyFieldProxy


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

    field_metadata_dict = MappingProxyType(
        {"static": static_props, "dynamic": dynamic_props, "event": dynamic_props}
    )
    base_attrs = {
        "__init__": person_record_proxy_init,
        "__field_metadata__": field_metadata_dict,
    }
    proxy_class_attrs = {**static_attrs, **dynamic_attrs, **event_attrs, **base_attrs}
    person_record_proxy_class = type("NumpyPersonRecordProxy", tuple(), proxy_class_attrs)
    return person_record_proxy_class
