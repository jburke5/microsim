from dataclasses import dataclass
from microsim.gender import NHANESGender
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.education import Education
from microsim.smoking_status import SmokingStatus


@dataclass
class BPCOGPersonStaticData:
    """Dataclass containing static data of a single BPCOG Person."""
    person_id: int
    gender: NHANESGender
    race_ethnicity: NHANESRaceEthnicity
    education: Education
    smoking_status: SmokingStatus
    random_effects_gcp: float
