class NumpyPopulationIterator:
    """Iterator for a whole population without conditions."""

    def __init__(self, person_store, at_t):
        self._person_store = person_store
        self._at_t = at_t
        self._range = iter(range(self._person_store._num_persons))

    def __iter__(self):
        return self

    def __next__(self):
        abs_person_idx = next(self._range)
        person_proxy = self._person_store.get_person_proxy_at(abs_person_idx, self._at_t)
        return person_proxy
