from microsim.store.numpy_population_iterator import NumpyPopulationIterator


class NumpyPopulationProxy:
    def __init__(self, person_store, at_t):
        self._person_store = person_store
        self._at_t = at_t

    def __iter__(self):
        return NumpyPopulationIterator(self._person_store, self._at_t)

    @property
    def num_persons(self):
        return self._person_store.get_num_persons()
