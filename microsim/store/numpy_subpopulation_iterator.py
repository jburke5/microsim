import numpy as np


class NumpySubpopulationIterator:
    """Iterator for a subset of a whole population, given by indices."""

    def __init__(self, person_store, at_t, member_indices, scratch_next=False):
        self._person_store = person_store
        self._at_t = at_t
        self._scratch_next = scratch_next
        self._it = np.nditer(member_indices, [], ["readonly"], [member_indices.dtype])

    def __iter__(self):
        return self

    def __next__(self):
        abs_person_idx = next(self._it)
        person_proxy = self._person_store.get_person_proxy_at(
            abs_person_idx, self._at_t, scratch_next=self._scratch_next
        )
        return person_proxy
