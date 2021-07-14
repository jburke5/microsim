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
            ("mi", [("type", np.unicode_, 9), ("fatal", np.bool_)]),
            ("stroke", [("type", np.unicode_, 9), ("fatal", np.bool_)]),
            ("dementia", [("type", np.unicode_, 9), ("fatal", np.bool_)]),
        ]
        return np.dtype(field_specs)

    def to_row_tuple(self, obj):
        if not isinstance(obj, BPCOGPersonEventRecord):
            raise TypeError(
                f"Given argument `obj` is not an instance of `BPCOGPersonEventRecord`: {obj}"
            )

        empty_string = ""  # Numpy converts None to 'None' for unicode dtypes, hence empty string
        has_mi = obj.mi is not None
        mi_type = obj.mi.type.value if has_mi else empty_string
        mi_fatal = obj.mi.fatal if has_mi else False

        has_stroke = obj.stroke is not None
        stroke_type = obj.stroke.type.value if has_stroke else empty_string
        stroke_fatal = obj.stroke.fatal if has_stroke else False

        has_dementia = obj.dementia is not None
        dementia_type = obj.dementia.type.value if has_dementia else empty_string
        dementia_fatal = obj.dementia.fatal if has_dementia else False

        values = (
            (mi_type, mi_fatal),
            (stroke_type, stroke_fatal),
            (dementia_type, dementia_fatal),
        )
        return values
