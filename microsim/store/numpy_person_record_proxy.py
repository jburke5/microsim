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


class PersonRecordProxyMetaclass(type):
    """
    Metaclass for automatically generating combined record proxies.

    Defines same properties on created classes as original record classes.
    """

    def __new__(cls, name, bases, namespace, *, field_metadata):
        static_props = field_metadata["static"]
        dynamic_props = field_metadata["dynamic"]
        event_props = field_metadata["event"]

        assert_unique_prop_names(static_props.keys(), dynamic_props.keys(), event_props.keys())

        static_attrs = proxy_attrs_from_props(static_props, "_static_row")
        dynamic_attrs = proxy_attrs_from_props(dynamic_props, "_dynamic_row")
        event_attrs = proxy_attrs_from_props(event_props, "_event_row")

        namespace.update(static_attrs)
        namespace.update(dynamic_attrs)
        namespace.update(event_attrs)

        namespace["__field_metadata__"] = field_metadata

        def person_record_proxy_init(self, static_row, dynamic_row, event_row):
            self._static_row = static_row
            self._dynamic_row = dynamic_row
            self._event_row = event_row

        namespace["__init__"] = person_record_proxy_init

        new_cls = type.__new__(cls, name, bases, namespace)
        return new_cls
