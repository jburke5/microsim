from microsim.store.numpy_person_proxy import NumpyPersonProxy
import numpy as np


class NumpyPopulationIterator:
    def __init__(
        self, static_data, dynamic_data, event_data, active_indices, new_person_record_proxy
    ):
        self._static_data = static_data
        self._dynamic_data = dynamic_data
        self._event_data = event_data
        self._new_person_record_proxy = new_person_record_proxy

        assert self._static_data.ndim == 1
        assert self._dynamic_data.ndim == 2
        assert self._event_data.ndim == 2

        assert self._dynamic_data.shape[0] == self._event_data.shape[0]
        assert (
            self._static_data.shape[0] == self._dynamic_data.shape[1] == self._event_data.shape[1]
        )

        self._num_ticks = self._dynamic_data.shape[0]
        self._it = np.nditer(active_indices, [], ["readonly"], [active_indices.dtype])

    def __iter__(self):
        return self

    def __next__(self):
        abs_person_idx = next(self._it)

        static_record = self._static_data[abs_person_idx]
        all_record_proxies = [
            self._new_person_record_proxy(
                static_record,
                self._dynamic_data[t][abs_person_idx],
                self._event_data[t][abs_person_idx],
            )
            for t in range(self._num_ticks)
        ]
        cur_proxy = all_record_proxies[-1]
        all_prev_proxies = all_record_proxies[:-1]
        person_proxy = NumpyPersonProxy(cur_proxy, all_prev_proxies)
        return person_proxy
