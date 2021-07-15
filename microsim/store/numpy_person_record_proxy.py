from itertools import chain, repeat


class NumpyPersonRecordProxy:
    """"""

    def __init__(
        self,
        static_row,
        dynamic_row,
        event_row,
        static_converter,
        dynamic_converter,
        event_converter,
    ):
        prop_name_to_row = dict(
            chain(
                zip(static_converter.get_property_names(), repeat(static_row)),
                zip(dynamic_converter.get_property_names(), repeat(dynamic_row)),
                zip(event_converter.get_property_names(), repeat(event_row)),
            )
        )
        object.__setattr__(self, "_prop_name_to_row", prop_name_to_row)

    def __getattribute__(self, name):
        row = object.__getattribute__(self, "_prop_name_to_row").get(name)
        if row is None:
            raise AttributeError(name)
        return row[name]

    def __setattr__(self, name, value):
        row = object.__getattribute__(self, "_prop_name_to_row").get(name)
        if row is None:
            raise AttributeError(name)
        row[name] = value
