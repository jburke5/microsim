class NumpyPopulationIterator:
    def __init__(self, person_store, at_t):
        self._store = person_store
        self._at_t = at_t
        self._range = range(self._num_persons)

    def __iter__(self):
        return self

    def __next__(self):
        abs_person_idx = next(self._range)
        person_proxy = self._person_store.get_person_proxy_at(
            abs_person_idx, self._at_t, scratch_next=self._scratch_next
        )
        return person_proxy
