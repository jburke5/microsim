from microsim.alcohol_category import AlcoholCategory
from microsim.person.base_person_record import BasePersonRecord
from microsim.gender import NHANESGender
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.education import Education
from microsim.smoking_status import SmokingStatus
from microsim.outcome import Outcome


class BPCOGPersonStaticRecord(BasePersonRecord):
    """Contains static data for one Person."""

    gender: NHANESGender
    raceEthnicity: NHANESRaceEthnicity
    education: Education
    smokingStatus: SmokingStatus
    randomEffectsGcp: float


class BPCOGPersonDynamicRecord(BasePersonRecord):
    """Contains dynamic data for one Person at one tick."""

    alive: bool
    age: int
    sbp: int
    dbp: int
    a1c: float
    hdl: int
    ldl: int
    trig: int
    totChol: int
    bmi: float
    waist: int
    anyPhysicalActivity: bool
    alcoholPerWeek: AlcoholCategory
    antiHypertensiveCount: int
    statin: int
    otherLipidLowerMedication: int
    bpMedsAdded: int
    afib: bool
    qalys: float
    gcp: float


class BPCOGPersonEventRecord(BasePersonRecord):
    """Contains events that happened to one Person during one tick."""

    mi: Outcome
    stroke: Outcome
    dementia: Outcome


class BPCOGPersonRecord(BPCOGPersonStaticRecord, BPCOGPersonDynamicRecord, BPCOGPersonEventRecord):
    """Contains all data for one Person during one tick."""

    pass
