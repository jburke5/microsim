import numpy as np
from microsim.store.numpy_person_record_proxy import NumpyPersonRecordProxy


class NumpyPopulationRecordProxy:
    def __init__(
        self,
        static_rows,
        dynamic_rows,
        event_rows,
        static_converter,
        dynamic_converter,
        event_converter,
        active_indices=None,
        active_condition=None,
    ):
        self._static_rows = static_rows
        self._dynamic_rows = dynamic_rows
        self._event_rows = event_rows
        self._static_converter = static_converter
        self._dynamic_converter = dynamic_converter
        self._event_converter = event_converter

        if active_indices is not None:
            self._active_indices = active_indices
        elif active_condition is not None:
            active_mask = self._unconditional_apply(active_condition, out_dtype=np.bool8)
            self._active_indices = active_mask.nonzero()
        else:
            raise ValueError("Expected to receive one of: `active_indices`, `active_condition`")

    @property
    def active_indices(self):
        return self._active_indices

    def _unconditional_apply(self, func, out_dtype=np.float64, **kwargs):
        """
        Applies `func` to each person record, then returns the result.

        Runs `func` even if person record is not active (i.e., even if its
        index does not appear in `self._active_indices`). Necessary to set
        `self._active_indices` in __init__ if given a condition, but may have
        other uses.
        """
        ops = [self._static_rows, self._dynamic_rows, self._event_rows, None]
        flags = []
        op_flags = [["readonly"], ["readonly"], ["readonly"], ["writeonly"]]
        op_dtypes = [
            self._static_rows.dtype,
            self._dynamic_rows.dtype,
            self._event_rows.dtype,
            out_dtype,
        ]
        with np.nditer(ops, flags, op_flags, op_dtypes) as it:
            for s, d, e, out in it:
                record_proxy = NumpyPersonRecordProxy(
                    s, d, e, self._static_converter, self._dynamic_converter, self._event_converter
                )
                out[...] = func(record_proxy, **kwargs)
            return it.operators[3]

    def apply(self, func, out_dtype=np.float64, **kwargs):
        """
        Applies `func` to each person record, then returns the result.

        Any keyword arguments that are not used will be passed to `func`.
        """
        ops = [self._active_indices, None]
        flags = []
        op_flags = [["readonly"], ["writeonly"]]
        op_dtypes = [
            self._active_indices.dtype,
            out_dtype,
        ]
        with np.nditer(ops, flags, op_flags, op_dtypes) as it:
            for i, out in it:
                static_record = self._static_rows[i]
                dynamic_record = self._dynamic_rows[i]
                event_record = self._dynamic_rows[i]
                record_proxy = NumpyPersonRecordProxy(
                    static_record,
                    dynamic_record,
                    event_record,
                    self._static_converter,
                    self._dynamic_converter,
                    self._event_converter,
                )
                out[...] = func(record_proxy, **kwargs)
            return it.operators[1]
