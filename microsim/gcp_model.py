import numpy as np
import pandas as pd
from microsim.gcp_outcome import GCPOutcome
from microsim.smoking_status import SmokingStatus
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.education import Education
from microsim.gender import NHANESGender
from microsim.person import Person
from collections import OrderedDict


class GCPModel:
    def __init__(self, outcomeModelRepository=None):
        #Q why are we doing this? I am not sure how/if this was used in the past
        self._outcome_model_repository = outcomeModelRepository
        pass

    def generate_next_outcome(self, person):
        fatal = False
        gcp = self.get_risk_for_person(person, person._rng)
        selfReported = False
        return GCPOutcome(fatal, selfReported, gcp)

    def get_next_outcome(self, person):
        return self.generate_next_outcome(person)

    #Q: I am not sure what the issue is here...
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
        #reportingDict = {}
        xb = 55.6090
        #reportingDict['intercept'] = xb
        xb += yearsInSim * -0.2031
        #reportingDict['yearsInSim'] = xb - pd.Series(reportingDict.values()).sum()
        if raceEthnicity == NHANESRaceEthnicity.NON_HISPANIC_BLACK:
            xb += -5.6818
            xb += yearsInSim * -0.00870
        #reportingDict['raceEthnicity'] = xb - pd.Series(reportingDict.values()).sum()
        if gender == NHANESGender.FEMALE:
            xb += 2.0863
            xb += yearsInSim * -0.06184
        #reportingDict['gender'] = xb - pd.Series(reportingDict.values()).sum()
        xb += -2.0109 * (baseAge - 65) / 10
        #reportingDict['baseAge'] = xb - pd.Series(reportingDict.values()).sum()
        xb += -0.1266 * yearsInSim * baseAge / 10
        #reportingDict['baseAgeYears'] = xb - pd.Series(reportingDict.values()).sum()
        # are we sure that the educatino categories align?
        if education == Education.LESSTHANHIGHSCHOOL:
            xb += -9.5559
        elif education == Education.SOMEHIGHSCHOOL:
            xb += -6.6495
        elif education == Education.HIGHSCHOOLGRADUATE:
            xb += -3.1954
        elif education == Education.SOMECOLLEGE:
            xb += -2.3795
        #reportingDict['educcation'] = xb - pd.Series(reportingDict.values()).sum()

        alcCoeffs = [0, 0.8071, 0.6943, 0.7706]
        xb += alcCoeffs[int(alcohol)]
        #reportingDict['alcohol'] = xb - pd.Series(reportingDict.values()).sum()

        if smokingStatus == SmokingStatus.CURRENT:
            xb += -1.1678
        #reportingDict['smoking'] = xb - pd.Series(reportingDict.values()).sum()
        xb += (bmi - 26.6) * 0.1309
        #reportingDict['bmi'] = xb - pd.Series(reportingDict.values()).sum()
        xb += (waist - 94) * -0.05754
        #reportingDict['waist'] = xb - pd.Series(reportingDict.values()).sum()
        # note...not 100% sure if this should be LDL vs. tot chol...
        xb += (totChol - 127) / 10 * 0.002690
        #reportingDict['totChol'] = xb - pd.Series(reportingDict.values()).sum()
        xb += (meanSBP - 120) / 10 * -0.2663
        #reportingDict['meanSbp'] = xb - pd.Series(reportingDict.values()).sum()
        xb += (meanSBP - 120) / 10 * yearsInSim * -0.01953
        #reportingDict['sbpYears'] = xb - pd.Series(reportingDict.values()).sum()

        xb += anyAntiHpertensive * 0.04410
        #reportingDict['antiHypertensive'] = xb - pd.Series(reportingDict.values()).sum()
        xb += anyAntiHpertensive * yearsInSim * 0.01984
        #reportingDict['antiHypertensiveYears'] = xb - pd.Series(reportingDict.values()).sum()

        # need to turn off the residual for hte simulation...also need to make sure that we're correctly centered...
        xb += (fastingGlucose - 100) / 10 * -0.09362
        #reportingDict['glucose'] = xb - pd.Series(reportingDict.values()).sum()
        if physicalActivity:
            xb += 0.6065
        #reportingDict['activity'] = xb - pd.Series(reportingDict.values()).sum()
        if afib:
            xb += -1.6579
        #reportingDict['afib'] = xb - pd.Series(reportingDict.values()).sum()
        #reportingDict['totalYears'] = yearsInSim
        #reportingDict['meanSBPValue'] = meanSBP
        #reportingDict['antiHypertensiveValue'] = anyAntiHpertensive
        #reportingDict['finalXb'] = xb

        #if self._outcome_model_repository is not None:
        #    self._outcome_model_repository.report_result('gcp', reportingDict)
        return xb

    def get_risk_for_person(self, person, rng=None, years=1, vectorized=False, test=False):
        if "gcp" not in list(person._randomEffects.keys()):
            person._randomEffects["gcp"] = person._rng.normal(0, 4.84)
        random_effect = person.gcpRandomEffect if vectorized else person._randomEffects["gcp"] 
        residual = 0 if test else rng.normal(0.38, 6.99)

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
                fastingGlucose=person.get_fasting_glucose(not test, rng),
                physicalActivity=person._anyPhysicalActivity[-1],
                afib=person._afib[-1],
            )

        return linPred + random_effect + residual
