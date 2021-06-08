from dataclasses import dataclass, fields
import numpy as np
from microsim.gender import NHANESGender
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.education import Education
from microsim.smoking_status import SmokingStatus
from microsim.person.pytype_to_nptype import pytype_to_nptype


@dataclass
class BPCOGPersonStaticData:
    """Dataclass containing static data of a single Person."""

    gender: NHANESGender
    raceEthnicity: NHANESRaceEthnicity
    education: Education
    smokingStatus: SmokingStatus
    randomEffectsGcp: float

    @property
    def dtype(self):
        field_specs = [(f.name, pytype_to_nptype(f.type)) for f in fields(self)]
        return np.dtype(field_specs)
