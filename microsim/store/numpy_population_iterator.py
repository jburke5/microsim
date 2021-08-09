from microsim.store.numpy_person_proxy import NumpyPersonProxy
import numpy as np


class NumpyPopulationIterator:
    def __init__(self, person_store, at_t, active_indices):
        self._person_store = person_store
        self._at_t = at_t
        self._it = np.nditer(active_indices, [], ["readonly"], [active_indices.dtype])

    def __iter__(self):
        return self

    def __next__(self):
        abs_person_idx = next(self._it)
        all_person_records = [
            self._person_store.get_person_record(abs_person_idx, t) for t in range(self._at_t + 1)
        ]
        cur_proxy = all_person_records[-1]
        all_prev_proxies = all_person_records[:-1]
        person_proxy = NumpyPersonProxy(cur_proxy, all_prev_proxies)
        return person_proxy
