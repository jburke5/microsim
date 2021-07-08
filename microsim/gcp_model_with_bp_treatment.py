import numpy as np
from microsim.smoking_status import SmokingStatus
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.education import Education
from microsim.gender import NHANESGender


class GCPModel:
    def __init__(self):
        pass

    # TODO — what do we need to do with the random intercept? shouls we take a draw per person and assign it?
    # if we don't do that there is going to be mroe change in cognitive trajectory per person that we'd expect...
    def calc_linear_predictor(self, person, test=False):
        xb = 55.6090
        xb += person.years_in_simulation() * -0.2031
        if person._raceEthnicity == NHANESRaceEthnicity.NON_HISPANIC_BLACK:
            xb += -5.6818
            xb += person.years_in_simulation() * -0.00870
        if person._gender == NHANESGender.FEMALE:
            xb += 2.0863
            xb += person.years_in_simulation() * -0.06184
        xb += -2.0109 * (person._age[0] - 65) / 10
        xb += -0.1266 * person.years_in_simulation() * (person._age[0] - 65) / 10

        # are we sure that the educatino categories align?
        if person._education == Education.LESSTHANHIGHSCHOOL:
            xb += -9.5559
        elif person._education == Education.SOMEHIGHSCHOOL:
            xb += -6.6495
        elif person._education == Education.HIGHSCHOOLGRADUATE:
            xb += -3.1954
        elif person._education == Education.SOMECOLLEGE:
            xb += -2.3795

        alcCoeffs = [0, 0.8071, 0.6943, 0.7706]
        xb += alcCoeffs[int(person._alcoholPerWeek[-1])]

        if person._smokingStatus == SmokingStatus.CURRENT:
            xb += -1.1678
        xb += (person._bmi[-1] - 26.6) * 0.1309
        xb += (person._waist[-1] - 94) * -0.05754
        # note...not 100% sure if this should be LDL vs. tot chol...
        xb += (person._totChol[-1] - 127) / 10 * 0.002690
        xb += (np.array(person._sbp).mean() - 120) / 10 * -0.2663
        xb += (np.array(person._sbp).mean() - 120) / 10 * person.years_in_simulation() * -0.01953

        xb += (person._antiHypertensiveCount[-1] > 0) * 0.04410
        xb += (person._antiHypertensiveCount[-1] > 0) * person.years_in_simulation() * 0.01984

        # need to turn off the residual for hte simulation...also need to make sure that we're correctly centered...
        xb += (person.get_fasting_glucose(not test) - 100) / 10 * -0.09362
        if person._anyPhysicalActivity[-1]:
            xb += 0.6065
        if person._afib[-1]:
            xb += -1.6579
        return xb

    def get_risk_for_person(self, person, years=1, test=False):
        random_effect = person._randomEffects["gcp"] if "gcp" in person._randomEffects else 0
        residual = 0 if test else np.random.normal(0.38, 6.99)
        return self.calc_linear_predictor(person, test) + random_effect + residual
