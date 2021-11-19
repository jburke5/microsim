from dataclasses import dataclass
from typing import Optional
from microsim.util.inherit_annotations import inherit_annotations
from typing_extensions import Protocol
from microsim.alcohol_category import AlcoholCategory
from microsim.gender import NHANESGender
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.education import Education
from microsim.smoking_status import SmokingStatus
from microsim.outcome import Outcome


class BPCOGPersonStaticRecordProtocol(Protocol):
    """Contains static data for one Person."""

    gender: NHANESGender
    raceEthnicity: NHANESRaceEthnicity
    education: Education
    smokingStatus: SmokingStatus
    gcpRandomEffect: float

    # props that likely should not be static in principle, yet do not currently change:
    otherLipidLowerMedication: int

    # props that should be with events in principle, but are best suited here for right now:
    selfReportMIAge: int = -9223372036854775808
    selfReportStrokeAge: int = -9223372036854775808


class BPCOGPersonDynamicRecordProtocol(Protocol):
    """Contains dynamic data for one Person at one tick."""

    alive: bool
    age: int
    sbp: float
    dbp: float
    a1c: float
    hdl: float
    ldl: float
    trig: float
    totChol: float
    bmi: float
    waist: float
    anyPhysicalActivity: bool
    alcoholPerWeek: AlcoholCategory
    antiHypertensiveCount: int
    statin: bool
    bpMedsAdded: int
    afib: bool
    qalys: float
    gcp: float


class BPCOGPersonEventRecordProtocol(Protocol):
    """Contains events that happened to one Person during one tick."""

    mi: Optional[Outcome]
    stroke: Optional[Outcome]
    dementia: Optional[Outcome]


class BPCOGPersonRecordProtocol(
    BPCOGPersonStaticRecordProtocol,
    BPCOGPersonDynamicRecordProtocol,
    BPCOGPersonEventRecordProtocol,
    Protocol,  # need to explicitly inherit from `Protocol` to make this class a `Protocol`
):
    """Contains all data for one Person during one tick."""

    pass


@dataclass
@inherit_annotations
class BPCOGPersonStaticRecord(BPCOGPersonStaticRecordProtocol):
    pass


@dataclass
@inherit_annotations
class BPCOGPersonDynamicRecord(BPCOGPersonDynamicRecordProtocol):
    pass


@dataclass
@inherit_annotations
class BPCOGPersonEventRecord(BPCOGPersonEventRecordProtocol):
    pass


@dataclass
@inherit_annotations
class BPCOGPersonRecord(BPCOGPersonRecordProtocol):
    pass
