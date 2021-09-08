from itertools import chain


def new_person_proxy_class(person_record_proxy_class):
    def person_proxy_init(self, next_record, cur_prev_records):
        self._next_record = next_record
        self._cur_prev_records = list(cur_prev_records)

    base_attrs = {
        "__init__": person_proxy_init,
        "current": property(lambda self: self._cur_prev_records[-1]),
        "current_and_previous": property(lambda self: self._cur_prev_records),
        "next": property(lambda self: self._next_record),
    }

    # proxy latest properties for vectorized model compatibility
    field_metadata = person_record_proxy_class.__field_metadata__
    all_record_prop_names = set(chain(*[c.keys() for c in field_metadata.values()]))

    prop_attrs = {}
    for prop_name in all_record_prop_names:
        prop_attrs[prop_name] = property(lambda self: getattr(self.current, prop_name))

    # add mean{prop_name} property for numeric dynamic fields
    for dynamic_prop_name in field_metadata["dynamic"].keys():
        mean_prop_name = f"mean{dynamic_prop_name.capitalize()}"
        prop_attrs[mean_prop_name] = property(
            lambda self: (
                sum([getattr(r, dynamic_prop_name) for r in self.current_and_previous])
                / len(self.current_and_previous)
            )
        )

    proxy_class_attrs = {**prop_attrs, **base_attrs}
    person_proxy_class = type("NumpyPersonProxy", tuple(), proxy_class_attrs)
    return person_proxy_class
