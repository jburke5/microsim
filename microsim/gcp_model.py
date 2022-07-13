import numpy as np
from microsim.smoking_status import SmokingStatus
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.education import Education
from microsim.gender import NHANESGender
from microsim.person import Person


class GCPModel:
    def __init__(self):
        pass

    # TODO — what do we need to do with the random intercept? shouls we take a draw per person and assign it?
    # if we don't do that there is going to be mroe change in cognitive trajectory per person that we'd expect...
    def calc_linear_predictor_for_patient_characteristics(
        self,
        yearsInSim,
        raceEthnicity,
        gender,
        baseAge,
        education,
        alcohol,
        smokingStatus,
        bmi,
        waist,
        totChol,
        meanSBP,
        anyAntiHpertensive,
        fastingGlucose,
        physicalActivity,
        afib,
        test=False,
    ):
        xb = 55.6090
        xb += yearsInSim * -0.2031
        if raceEthnicity == NHANESRaceEthnicity.NON_HISPANIC_BLACK:
            xb += -5.6818
            xb += yearsInSim * -0.00870
        if gender == NHANESGender.FEMALE:
            xb += 2.0863
            xb += yearsInSim * -0.06184
        xb += -2.0109 * (baseAge - 65) / 10
        xb += -0.1266 * yearsInSim * baseAge / 10

        # are we sure that the educatino categories align?
        if education == Education.LESSTHANHIGHSCHOOL:
            xb += -9.5559
        elif education == Education.SOMEHIGHSCHOOL:
            xb += -6.6495
        elif education == Education.HIGHSCHOOLGRADUATE:
            xb += -3.1954
        elif education == Education.SOMECOLLEGE:
            xb += -2.3795

        alcCoeffs = [0, 0.8071, 0.6943, 0.7706]
        xb += alcCoeffs[int(alcohol)]

        if smokingStatus == SmokingStatus.CURRENT:
            xb += -1.1678
        xb += (bmi - 26.6) * 0.1309
        xb += (waist - 94) * -0.05754
        # note...not 100% sure if this should be LDL vs. tot chol...
        xb += (totChol - 127) / 10 * 0.002690
        xb += (meanSBP - 120) / 10 * -0.2663
        xb += (meanSBP - 120) / 10 * yearsInSim * -0.01953

        xb += anyAntiHpertensive * 0.04410
        xb += anyAntiHpertensive * yearsInSim * 0.01984

        # need to turn off the residual for hte simulation...also need to make sure that we're correctly centered...
        xb += (fastingGlucose - 100) / 10 * -0.09362
        if physicalActivity:
            xb += 0.6065
        if afib:
            xb += -1.6579
        return xb

    def get_risk_for_person(self, person, years=1, vectorized=False, test=False):
        random_effect = 0
        if not vectorized:
            random_effect = person._randomEffects["gcp"] if "gcp" in person._randomEffects else 0
        residual = 0 if test else np.random.normal(0.38, 6.99)

        linPred = 0
        if vectorized:
            linPred = self.calc_linear_predictor_for_patient_characteristics(
                yearsInSim=person.totalYearsInSim,
                raceEthnicity=person.raceEthnicity,
                gender=person.gender,
                baseAge=person.baseAge,
                education=person.education,
                alcohol=person.alcoholPerWeek,
                smokingStatus=person.smokingStatus,
                bmi=person.bmi,
                waist=person.waist,
                totChol=person.totChol,
                meanSBP=person.meanSbp,
                anyAntiHpertensive=((person.antiHypertensiveCount + person.totalBPMedsAdded)> 0),
                fastingGlucose=Person.convert_a1c_to_fasting_glucose(person.a1c),
                physicalActivity=person.anyPhysicalActivity,
                afib=person.afib,
            )
        else:
            linPred = self.calc_linear_predictor_for_patient_characteristics(
                yearsInSim=person.years_in_simulation(),
                raceEthnicity=person._raceEthnicity,
                gender=person._gender,
                baseAge=person._age[0],
                education=person._education,
                alcohol=person._alcoholPerWeek[-1],
                smokingStatus=person._smokingStatus,
                bmi=person._bmi[-1],
                waist=person._waist[-1],
                totChol=person._totChol[-1],
                meanSBP=np.array(person._sbp).mean(),
                anyAntiHpertensive=((person._antiHypertensiveCount[-1] + np.array(person._bpMedsAdded).sum()) > 0),
                fastingGlucose=person.get_fasting_glucose(not test),
                physicalActivity=person._anyPhysicalActivity[-1],
                afib=person._afib[-1],
            )

        return linPred + random_effect + residual
