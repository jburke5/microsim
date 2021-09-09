from itertools import chain


class DirectProxiedProperty:
    def __init__(self, proxied_prop_name):
        self._proxied_prop_name = proxied_prop_name

    def __get__(self, instance, owner=None):
        cur_record = instance.current
        value = getattr(cur_record, self._proxied_prop_name)
        return value


class MeanProxiedProperty:
    def __init__(self, proxied_prop_name):
        self._proxied_prop_name = proxied_prop_name

    def __get__(self, instance, owner=None):
        cur_prev_records = instance.current_and_previous
        values = [getattr(r, self._proxied_prop_name) for r in cur_prev_records]
        mean = sum(values) / len(cur_prev_records)
        return mean


def new_person_proxy_class(field_metadata):
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
    all_record_prop_names = set(chain(*[c.keys() for c in field_metadata.values()]))

    prop_attrs = {}
    for prop_name in all_record_prop_names:
        prop_attrs[prop_name] = DirectProxiedProperty(prop_name)

    # add mean{prop_name} property for numeric dynamic fields
    for dynamic_prop_name in field_metadata["dynamic"].keys():
        mean_prop_name = f"mean{dynamic_prop_name.capitalize()}"
        prop_attrs[mean_prop_name] = MeanProxiedProperty(dynamic_prop_name)

    proxy_class_attrs = {**prop_attrs, **base_attrs}
    person_proxy_class = type("NumpyPersonProxy", tuple(), proxy_class_attrs)
    return person_proxy_class


class NumpyPersonProxy:
    def __init__(self, next_record, cur_prev_records):
        self._next_record = next_record
        self._cur_prev_records = list(cur_prev_records)

    @property
    def next(self):
        return self._next_record

    @property
    def current(self):
        return self._cur_prev_records[-1]

    @property
    def current_and_previous(self):
        return self._cur_prev_records
