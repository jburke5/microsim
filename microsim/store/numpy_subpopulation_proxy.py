import numpy as np
from microsim.store.numpy_subpopulation_iterator import NumpySubpopulationIterator


class NumpySubpopulationProxy:
    def __init__(self, person_store, at_t, member_indices, scratch_next=False):
        self._at_t = at_t
        self._person_store = person_store
        self._scratch_next = scratch_next

        index_array = np.array(member_indices, dtype=np.intp)
        if index_array.ndim != 1:
            raise ValueError(
                "Expected `member_indices` to convert to 1D array;"
                f" converted to {index_array.ndim}D instead (from value: {member_indices})"
            )
        self._member_indices = index_array

    @property
    def member_indices(self):
        return self._member_indices

    @property
    def num_persons(self):
        return self._member_indices.shape[0]

    def __iter__(self):
        return NumpySubpopulationIterator(self._person_store, self._at_t, self._member_indices)

    def __getitem__(self, key):
        if type(key) is not int:
            raise TypeError(f"Expected `int` key; received: {type(key)} (value: '{key}')")

        abs_idx = self._member_indices[key]
        person_proxy = self._person_store.get_person_proxy_at(
            abs_idx, self._at_t, scratch_next=self._scratch_next
        )
        return person_proxy

    def get_scratch_copy(self):
        """Returns subpopulation with same members but with scratch `.next`."""
        return NumpySubpopulationProxy(
            self._person_store, self._at_t, self._member_indices, scratch_next=True
        )
