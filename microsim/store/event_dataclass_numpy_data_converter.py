import numpy as np
from microsim.person.bpcog_person_records import BPCOGPersonEventRecord
from microsim.store.base_numpy_data_converter import BaseNumpyDataConverter


class EventDataclassNumpyDataConverter(BaseNumpyDataConverter):
    """
    Converts Event dataclasses into Numpy array rows.

    (WIP) Tightly coupled with BPCOGPersonEventRecord while the current
    EventRecord and Outcome classes are out of alignment with the flat array
    structure of NumpyPersonStore.
    """

    def get_dtype(self):
        field_specs = [
            ("has_mi": np.bool_),
            ("mi_fatal", np.bool_),
            ("has_stroke", np.bool_),
            ("stroke_fatal", np.bool_),
            ("has_dementia", np.bool_),
        ]
        return np.dtype(field_specs)

    def to_row_tuple(self, obj):
        if not isinstance(obj, BPCOGPersonEventRecord):
            raise TypeError(
                f"Given argument `obj` is not an instance of `BPCOGPersonEventRecord`: {obj}"
            )

        has_mi = obj.mi is not None
        mi_fatal = obj.mi.fatal if has_mi else False

        has_stroke = obj.stroke is not None
        stroke_fatal = obj.stroke.fatal if has_stroke else False

        has_dementia = obj.dementia is not None

        values = (
            has_mi,
            mi_fatal,
            has_stroke,
            stroke_fatal,
            has_dementia,
        )
        return values
