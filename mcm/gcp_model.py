import numpy as np
from mcm.smoking_status import SmokingStatus
from mcm.race_ethnicity import NHANESRaceEthnicity
from mcm.education import Education
from mcm.gender import NHANESGender


class GCPModel:

    def __init__(self):
        pass

    # TODO..make sure centering is right for all variables...will fail hard on validation
    # TODO — what do we need to do with the random intercept? shouls we take a draw per person and assign it?
    # if we don't do that there is going to be mroe change in cognitive trajectory per person that we'd expect...
    def calc_linear_predictor(self, person):
        xb = 55.6090
        xb += person.years_in_simulation() * -0.2031
        if person._raceEthnicity == NHANESRaceEthnicity.NON_HISPANIC_BLACK:
            xb += -5.6818
            xb += person.years_in_simulation() * -0.00870
        if person._gender == NHANESGender.FEMALE:
            xb += 2.0863
            xb += person.years_in_simulation() * -0.06184
        xb += -2.0109 * person._age[0]/10
        xb += -0.1266 * person.years_in_simulation() * person._age[0]/10
        if person._education == Education.LESSTHANHIGHSCHOOL:
            xb += -9.5559
        elif person._education == Education.SOMEHIGHSCHOOL:
            xb += -6.6495
        elif person._education == Education.HIGHSCHOOLGRADUATE:
            xb += -3.1954
        elif person._education == Education.SOMECOLLEGE:
            xb += -2.3795

        # TODO...figure otu what to do with A1cs
        # alc1	0.8071
        # alc2	0.6943
        # alc3	0.7706
        if person._smokingStatus == SmokingStatus.CURRENT:
            xb += -1.1678
        xb += person._bmi[-1] * 0.1309
        xb += person._waist[-1] * -0.05754
        xb += person._totChol[-1]/10 * 0.002690
        xb += (np.array(person._sbp).mean()-120) * -0.2663
        xb += (np.array(person._sbp).mean()-120) * person.years_in_simulation() * -0.01953

        # need to figure otu what to do with glucose
        # gluc10 - 0.09362
        if person._anyPhysicalActivity[-1]:
            xb += 0.6065
        if person._afib[-1]:
            xb += -1.6579
        return xb

    # TODO : need to add some tests cases to make sure this syncs up
    # TODO : need to account for uyncertainty...random draws from residual distrribution +/- accounting for coefficient variation
    def get_risk_for_person(self, person, years=1):
        return self.calc_linear_predictor(person)
