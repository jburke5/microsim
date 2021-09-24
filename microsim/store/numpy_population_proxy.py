from microsim.store.numpy_population_iterator import NumpyPopulationIterator
import numpy as np


class NumpyPopulationProxy:
    def __init__(
        self, person_store, at_t, active_indices=None, active_condition=None, scratch_next=False
    ):
        self._at_t = at_t
        self._person_store = person_store
        self._scratch_next = scratch_next

        if active_indices is not None:
            self._active_indices = active_indices
        elif active_condition is not None:
            active_mask = self._get_active_mask(active_condition)
            (self._active_indices,) = active_mask.nonzero()
        else:
            raise ValueError("Expected to receive one of: `active_indices`, `active_condition`")

    @property
    def active_indices(self):
        return self._active_indices

    @property
    def num_persons(self):
        return self._active_indices.shape[0]

    def __iter__(self):
        return NumpyPopulationIterator(self._person_store, self._at_t, self._active_indices)

    def get_scratch_copy(self):
        """Returns population with same members but with scratch as `.next`."""
        return NumpyPopulationProxy(
            self._person_store, self._at_t, self._active_indices, scratch_next=True
        )

    def _get_active_mask(self, active_condition):
        num_persons = self._person_store.get_num_persons()
        all_person_indices = np.arange(num_persons)
        active_mask = np.zeros(all_person_indices.shape, dtype=np.bool8)

        ops = [all_person_indices, active_mask]
        flags = []
        op_flags = [["readonly"], ["writeonly"]]
        op_dtypes = [all_person_indices.dtype, active_mask.dtype]
        with np.nditer(ops, flags, op_flags, op_dtypes) as it:
            for i, out in it:
                person_record = self._person_store.get_person_record(i, self._at_t)
                is_active = active_condition(person_record)
                out[...] = is_active
            return it.operands[1]
