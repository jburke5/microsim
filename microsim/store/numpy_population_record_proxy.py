from microsim.store.numpy_person_record_proxy import NumpyPersonRecordProxy
import numpy as np


class NumpyPopulationRecordProxy:
    def __init__(
        self,
        static_rows,
        dynamic_rows,
        event_rows,
        static_converter,
        dynamic_converter,
        event_converter,
    ):
        self._static_rows = static_rows
        self._dynamic_rows = dynamic_rows
        self._event_rows = event_rows
        self._static_converter = static_converter
        self._dynamic_converter = dynamic_converter
        self._event_converter = event_converter

    def apply(self, func, **kwargs):
        """
        Applies `func` to each person record, then returns the result.

        Any keyword arguments that are not used will be passed to `func`.
        """
        ops = [self._static_rows, self._dynamic_rows, self._event_rows, None]
        flags = []
        op_flags = [["readonly"], ["readonly"], ["readonly"], ["writeonly"]]
        op_dtypes = [
            self._static_converter.get_dtype(),
            self._dynamic_converter.get_dtype(),
            self._event_converter.get_dtype(),
            np.float64,
        ]
        with np.nditer(ops, flags, op_flags, op_dtypes) as it:
            for s, d, e, out in it:
                record_proxy = NumpyPersonRecordProxy(
                    s, d, e, self._static_converter, self._dynamic_converter, self._event_converter
                )
                out[...] = func(record_proxy, **kwargs)
            return out
