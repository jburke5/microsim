from microsim.person import Person


def person_obj_from_person_record(population_index, person_record):
    person = Person(
        age=person_record.age,
        gender=person_record.gender,
        raceEthnicity=person_record.raceEthnicity,
        sbp=person_record.sbp,
        dbp=person_record.dbp,
        a1c=person_record.a1c,
        hdl=person_record.hdl,
        totChol=person_record.totChol,
        bmi=person_record.bmi,
        ldl=person_record.ldl,
        trig=person_record.trig,
        waist=person_record.waist,
        anyPhysicalActivity=person_record.anyPhysicalActivity,
        education=person_record.education,
        smokingStatus=person_record.smokingStatus,
        alcohol=person_record.alcoholPerWeek,
        antiHypertensiveCount=person_record.antiHypertensiveCount,
        statin=person_record.statin,
        otherLipidLoweringMedicationCount=person_record.otherLipidLowerMedication,
        initializeAfib=lambda _: person_record.afib,
        selfReportStrokeAge=person_record.selfReportStrokeAge,
        selfReportMIAge=person_record.selfReportMIAge,
        randomEffects={"gcp": person_record.gcpRandomEffect},
        _populationIndex=population_index,
    )
    person._qalys.append(person_record.qalys)
    person._gcp.append(person_record.gcp)
    return person
