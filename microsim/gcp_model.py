import numpy as np
from microsim.person import Person
from microsim.smoking_status import SmokingStatus
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.education import Education
from microsim.gender import NHANESGender


class GCPModel:
    def __init__(self):
        pass

    def calc_linear_predictor(self, person):
        return self.calc_linear_predictor_for_patient_characteristics(
            years_in_simulation=person.years_in_simulation(),
            raceEthnicity=person._raceEthnicity,
            gender=person._gender,
            baseAge=person._age[0],
            education=person._education,
            smokingStatus=person._smokingStatus,
            bmi=person._bmi[-1],
            waist=person._waist[-1],
            totChol=person._totChol[-1],
            meanSbp=np.array(person._sbp).mean(),
            afib=person._afib[-1],
            anyPhysicalActivity=person._anyPhysicalActivity[-1],
            alc=person._alcoholPerWeek[-1],
            antiHypertensiveCount=person._antiHypertensiveCount[-1],
            a1c=person._a1c[-1],
        )

    def calc_linear_predictor_vectorized(self, x):
        return self.calc_linear_predictor_for_patient_characteristics(
            years_in_simulation=x.totalYearsInSim,
            raceEthnicity=x.raceEthnicity,
            gender=x.gender,
            baseAge=x.baseAge,
            education=x.education,
            smokingStatus=x.smokingStatus,
            bmi=x.bmi,
            waist=x.waist,
            totChol=x.totChol,
            meanSbp=x.meanSbp,
            afib=x.afib,
            anyPhysicalActivity=x.anyPhysicalActivity,
            alc=x.alcoholPerWeek,
            antiHypertensiveCount=x.antiHypertensiveCount,
            a1c=x.a1c,
        )

    def calc_linear_predictor_for_patient_characteristics(
        self,
        years_in_simulation,
        raceEthnicity,
        gender,
        baseAge,
        education,
        smokingStatus,
        bmi,
        waist,
        totChol,
        meanSbp,
        afib,
        anyPhysicalActivity,
        alc,
        antiHypertensiveCount,
        a1c,
    ):
        xb = 55.6090
        xb += years_in_simulation * -0.2031
        if raceEthnicity == NHANESRaceEthnicity.NON_HISPANIC_BLACK:
            xb += -5.6818
            xb += years_in_simulation * -0.00870
        if gender == NHANESGender.FEMALE:
            xb += 2.0863
            xb += years_in_simulation * -0.06184
        xb += -2.0109 * baseAge / 10
        xb += -0.1266 * years_in_simulation * baseAge / 10
        if education == Education.LESSTHANHIGHSCHOOL:
            xb += -9.5559
        elif education == Education.SOMEHIGHSCHOOL:
            xb += -6.6495
        elif education == Education.HIGHSCHOOLGRADUATE:
            xb += -3.1954
        elif education == Education.SOMECOLLEGE:
            xb += -2.3795

        alcCoeffs = [0, 0.8071, 0.6943, 0.7706]
        xb += alcCoeffs[int(alc)]

        if smokingStatus == SmokingStatus.CURRENT:
            xb += -1.1678
        xb += bmi * 0.1309
        xb += waist * -0.05754
        xb += totChol / 10 * 0.002690
        xb += (meanSbp - 120) * -0.2663
        xb += (meanSbp - 120) * years_in_simulation * -0.01953

        xb += (antiHypertensiveCount > 0) * 0.04410
        xb += (antiHypertensiveCount > 0) * years_in_simulation * 0.01984

        xb += (Person.convert_a1c_to_fasting_glucose(a1c) - 100) / 10 * -0.09362

        if anyPhysicalActivity:
            xb += 0.6065
        if afib:
            xb += -1.6579
        return xb

    # TODO : need to account for uyncertainty...random draws from residual distrribution +/- accounting for coefficient variation
    def get_risk_for_person(self, person, years=1, vectorized=False):
        return (
            self.calc_linear_predictor_vectorized(person)
            if vectorized
            else self.calc_linear_predictor(person)
        )
