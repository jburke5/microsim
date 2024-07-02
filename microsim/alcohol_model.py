import numpy as np

from microsim.alcohol_category import AlcoholCategory
from microsim.smoking_status import SmokingStatus
from microsim.race_ethnicity import RaceEthnicity
from microsim.gender import NHANESGender

class AlcoholPrevalenceModel:

    def __init__(self):
        pass

    def calc_linear_predictor_for_patient_characteristics(self, gender, smokingStatus, age):
        """Returns 3 linear predictors for the four alcohol levels.
        These 3 linear predictors correspond to the first 3 cumulative probabilities of the first 3 alcohol levels.
        The 4th cumulative probability is exactly 1 since it includes all alcohol levels.
        Each alcohol level has its own intercept, but the coefficient of each factor is constant across alcohol levels.
        This is an ordered logistic regression model.
        Based on NHANES data (1999-2017) and the polr package in r.
        This model was built in order to initialize person objects when alcohol level is missing."""
        xb = 0

        if gender==NHANESGender.MALE:
            xb += 0.
        elif gender==NHANESGender.FEMALE:
            xb += -(-0.85537)
        else:
            raise RuntimeError("Unknown raceEthnicity in EducationPrevalenceModel.")

        if smokingStatus==SmokingStatus.NEVER:
            xb += 0
        elif smokingStatus==SmokingStatus.FORMER:
            xb += -(0.82691)
        elif smokingStatus==SmokingStatus.CURRENT:
            xb += -(1.48850)
        else:
            raise RuntimeError("Unknown smokingStatus in EducationPrevalenceModel.")

        xb += -(-0.02906)*age
        #intercepts in polr package results are the actual intercepts
        lps = (xb-3.1956, xb-3.1956, xb-1.5056)
        return lps

    def estimate_next_risk(self, person):
        linearPredictors = self.calc_linear_predictor_for_patient_characteristics(person._gender, person._smokingStatus, person._age[-1])
        #cumulative probabilities for the first 3 alcohol levels
        [cp1, cp2, cp3] = list(map(lambda x: self.inv_logit(x), linearPredictors))
        draw = person._rng.uniform()
        if draw<cp1:
            return AlcoholCategory.NONE
        elif draw<cp2:
            return AlcoholCategory.ONETOSIX
        elif draw<cp3:
            return AlcoholCategory.SEVENTOTHIRTEEN
        elif draw<1.:
            return AlcoholCategory.FOURTEENORMORE
        else:
             raise RuntimeError("Draw not consistent with cumulative probabilities in AlcoholPrevalenceModel.estimate_next_risk.")

    def inv_logit(self, lp):
        # note: limit the calculation to avoid over/under-flow issues
        if lp<-10:
            risk = 0.
        elif lp>10.:
            risk = 1.
        else:
            risk = 1/(1+np.exp(-lp))
        return risk

