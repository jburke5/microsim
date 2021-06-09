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

    @classmethod
    def get_numpy_dtype(cls):
        """Returns np.dtype that numpy can use to represent this class' data."""
        field_specs = [(f.name, pytype_to_nptype(f.type)) for f in fields(cls)]
        return np.dtype(field_specs)

    def as_numpy_arraylike(self):
        """
        Returns a tuple that, along with `get_numpy_dtype()`, numpy can use to create an array.
        """
        dtype = self.get_numpy_dtype()
        values = tuple(getattr(self, name) for name in dtype.names)
        return values
