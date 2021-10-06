from collections import defaultdict
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

    def where(self, condition):
        """Returns subpopulation of persons that satisfy the given condition."""
        subpop_indices = [i for i, person in enumerate(self) if condition(person)]
        subpop = NumpySubpopulationProxy(self._person_store, self._at_t, subpop_indices)
        return subpop

    def group_by(self, key_func):
        """Groups persons into subpopulations per the given `key_func`."""
        group_indices = defaultdict(list)
        for i, person in enumerate(self):
            person_group_key = key_func(person)
            group_indices[person_group_key].append(i)

        grouped_subpops = {
            group_key: NumpySubpopulationProxy(self._person_store, self._at_t, subpop_indices)
            for group_key, subpop_indices in group_indices.items()
        }
        return grouped_subpops
