from dataclasses import dataclass
from microsim.gender import NHANESGender
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.education import Education
from microsim.smoking_status import SmokingStatus


@dataclass
class BPCOGPersonStaticData:
    """Dataclass containing static data of a single Person."""

    gender: NHANESGender
    raceEthnicity: NHANESRaceEthnicity
    education: Education
    smokingStatus: SmokingStatus
    randomEffectsGcp: float
