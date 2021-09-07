from itertools import chain


class NumpyPersonProxy:
    def __init__(self, at_t, next_record, cur_prev_records):
        self._at_t = at_t
        self._next_record = next_record
        self._cur_prev_records = list(cur_prev_records)

        # proxy latest properties for vectorized model compatibility
        num_cur_prev_records = len(self._cur_prev_records)
        if self._at_t == -1:
            # ...though don't proxy properties for initial data load
            return

        cur_record = self._cur_prev_records[-1]
        field_metadata = cur_record.__field_metadata__
        all_prop_names = set(chain(*[c.keys() for c in field_metadata.values()]))

        for prop_name in all_prop_names:
            setattr(self, prop_name, getattr(cur_record, prop_name))

        # add mean{prop_name} property for numeric dynamic fields
        for prop_name in field_metadata["dynamic"].keys():
            mean_prop_name = f"mean{prop_name.capitalize()}"
            mean_val = (
                sum([getattr(r, prop_name) for r in self._cur_prev_records]) / num_cur_prev_records
            )
            setattr(self, mean_prop_name, mean_val)

    @property
    def next(self):
        return self._next_record

    @property
    def current(self):
        return self._cur_prev_records[-1]

    @property
    def current_and_previous(self):
        return self._cur_prev_records
