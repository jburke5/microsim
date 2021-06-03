from dataclasses import dataclass
from microsim.alcohol_category import AlcoholCategory


@dataclass
class BPCOGPersonDynamicData:
    """Dataclass containing dynamic data of a single Person during one tick."""

    personId: int
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
