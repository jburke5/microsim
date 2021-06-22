from dataclasses import dataclass, fields
import numpy as np
from microsim.gender import NHANESGender
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.education import Education
from microsim.smoking_status import SmokingStatus
from microsim.person.pytype_to_nptype import pytype_to_nptype
from microsim.store.numpy_data_converter_protocol import NumpyDataConverterProtocol


@dataclass
class BPCOGPersonStaticData(NumpyDataConverterProtocol["BPCOGPersonStaticData"]):
    """Dataclass containing static data of a single Person."""

    gender: NHANESGender
    raceEthnicity: NHANESRaceEthnicity
    education: Education
    smokingStatus: SmokingStatus
    randomEffectsGcp: float

    def get_dtype(self):
        field_specs = [(f.name, pytype_to_nptype(f.type)) for f in fields(self)]
        return np.dtype(field_specs)

    def to_row_tuple(self, obj: "BPCOGPersonStaticData"):
        dtype = obj.get_dtype()
        values = tuple(getattr(self, name) for name in dtype.names)
        return values
