from typing import Callable

import numpy as np
from microsim.store.numpy_person_proxy import NumpyPersonProxy
from microsim.store.numpy_population_iterator import NumpyPopulationIterator
from microsim.store.numpy_subpopulation_proxy import NumpySubpopulationProxy


class NumpyPopulationProxy:
    def __init__(self, person_store, at_t):
        self._person_store = person_store
        self._at_t = at_t

    def __iter__(self):
        return NumpyPopulationIterator(self._person_store, self._at_t)

    @property
    def num_persons(self):
        return self._person_store.get_num_persons()

    def where(self, condition: Callable[[NumpyPersonProxy], bool]):
        """Returns subpopulation of persons that satisfy the given condition."""
        subpop_indices = np.array(
            [i for i, person in enumerate(self) if condition(person)],
            dtype=np.intp,
        )
        subpop = NumpySubpopulationProxy(self._person_store, self._at_t, subpop_indices)
        return subpop
