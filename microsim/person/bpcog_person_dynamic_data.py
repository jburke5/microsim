from dataclasses import dataclass
from microsim.alcohol_category import AlcoholCategory


@dataclass
class BPCOGPersonDynamicData:
    """Dataclass containing dynamic data of a single Person during one tick."""
    person_id: int
    alive: bool
    age: int
    sbp: int
    dbp: int
    a1c: float
    hdl: int
    ldl: int
    trig: int
    tot_chol: int
    bmi: float
    waist: int
    any_physical_activity: bool
    alcohol_per_week: AlcoholCategory
    anti_hypertensive_count: int
    statin: int
    other_lipid_lower_medication: int
    bp_meds_added: int
    afib: bool
    qalys: float
    gcp: float
