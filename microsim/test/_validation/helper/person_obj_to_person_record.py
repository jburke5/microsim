from microsim.alcohol_category import AlcoholCategory
from microsim.education import Education
from microsim.gender import NHANESGender
from microsim.person.bpcog_person_records import BPCOGPersonRecord
from microsim.outcome import OutcomeType
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.smoking_status import SmokingStatus


def person_obj_to_person_record(person, at_t):
    mi_at_t = [o for t, o in person._outcomes[OutcomeType.MI] if t == at_t]
    assert len(mi_at_t) == 0 or len(mi_at_t) == 1 and mi_at_t[0].type == OutcomeType.MI
    mi = mi_at_t[0] if mi_at_t else None

    stroke_at_t = [o for t, o in person._outcomes[OutcomeType.STROKE] if t == at_t]
    assert len(stroke_at_t) == 0 or (
        len(stroke_at_t) == 1 and stroke_at_t[0].type == OutcomeType.STROKE
    )
    stroke = stroke_at_t[0] if stroke_at_t else None

    dementia_at_t = [o for t, o in person._outcomes[OutcomeType.DEMENTIA] if t == at_t]
    assert len(dementia_at_t) == 0 or (
        len(dementia_at_t) == 1 and dementia_at_t[0].type == OutcomeType.DEMENTIA
    )
    dementia = dementia_at_t[0] if dementia_at_t else None

    # `advance_vectorized` workarounds:
    # `alive` only updated when person dies
    is_alive = person._alive[-1]
    # `age` not updated if person dies that tick
    age = person._age[at_t] if is_alive else person._age[at_t - 1] + 1
    # `otherLipidLoweringMedicationCount` never updated after init? (unclear if intentional)
    otherLipidLowerMedication = person._otherLipidLoweringMedicationCount[-1]

    selfReportAgeKwargs = {}
    if hasattr(person, "_selfReportStrokeAge"):
        selfReportAgeKwargs["selfReportStrokeAge"] = person._selfReportStrokeAge
    if hasattr(person, "_selfReportMIAge"):
        selfReportAgeKwargs["selfReportMIAge"] = person._selfReportMIAge

    person_record = BPCOGPersonRecord(
        gender=NHANESGender(person._gender),
        raceEthnicity=NHANESRaceEthnicity(person._raceEthnicity),
        education=Education(person._education),
        smokingStatus=SmokingStatus(person._smokingStatus),
        gcpRandomEffect=person._randomEffects["gcp"],
        alive=is_alive,
        age=age,
        sbp=person._sbp[at_t],
        dbp=person._dbp[at_t],
        a1c=person._a1c[at_t],
        hdl=person._hdl[at_t],
        ldl=person._ldl[at_t],
        trig=person._trig[at_t],
        totChol=person._totChol[at_t],
        bmi=person._bmi[at_t],
        waist=person._waist[at_t],
        anyPhysicalActivity=person._anyPhysicalActivity[at_t],
        alcoholPerWeek=AlcoholCategory(person._alcoholPerWeek[at_t]),
        antiHypertensiveCount=person._antiHypertensiveCount[at_t],
        statin=person._statin[at_t],
        otherLipidLowerMedication=otherLipidLowerMedication,
        bpMedsAdded=person._bpMedsAdded[at_t],
        afib=person._afib[at_t],
        qalys=person._qalys[at_t],
        gcp=person._gcp[at_t],
        mi=mi,
        stroke=stroke,
        dementia=dementia,
        **selfReportAgeKwargs,
    )

    return person_record
