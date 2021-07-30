import numpy as np


class NumpyPopulationIterator:
    def __init__(
        self, static_rows, dynamic_rows, event_rows, active_indices, new_person_record_proxy
    ):
        self._static_rows = static_rows
        self._dynamic_rows = dynamic_rows
        self._event_rows = event_rows
        self._new_person_record_proxy = new_person_record_proxy

        self._it = np.nditer(active_indices, [], ["readonly"], [active_indices.dtype])

    def __iter__(self):
        return self

    def __next__(self):
        abs_idx = next(self._it)
        static_record = self._static_rows[abs_idx]
        dynamic_record = self._dynamic_rows[abs_idx]
        event_record = self._event_rows[abs_idx]
        person_record_proxy = self._new_person_record_proxy(
            static_record, dynamic_record, event_record
        )
        return person_record_proxy
