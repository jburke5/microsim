import numpy as np

from microsim.education import Education
from microsim.smoking_status import SmokingStatus
from microsim.race_ethnicity import RaceEthnicity

class EducationPrevalenceModel:

    def __init__(self):
        pass

    def calc_linear_predictor_for_patient_characteristics(self, raceEthnicity, smokingStatus, age):
        """Returns 4 linear predictors for the five education levels.
        These 4 linear predictors correspond to the first 4 cumulative probabilities of the first 4 education levels.
        The 5th cumulative probability is exactly 1 since it includes all education levels.
        Each education level has its own intercept, but the coefficient of each factor is constant across education levels.
        This is an ordered logistic regression model.
        Based on NHANES data (1999-2017) and the polr package in r.
        This model was built in order to initialize person objects when education level is missing."""
        xb = 0
 
        if raceEthnicity==RaceEthnicity.MEXICAN_AMERICAN:
            xb += 0.
        elif raceEthnicity==RaceEthnicity.OTHER_HISPANIC:
            xb += -0.84732
        elif (raceEthnicity==RaceEthnicity.NON_HISPANIC_WHITE) | (raceEthnicity==RaceEthnicity.ASIAN):
            xb += -1.99639
        elif raceEthnicity==RaceEthnicity.NON_HISPANIC_BLACK:
            xb += -1.23381
        elif raceEthnicity==RaceEthnicity.OTHER:
            xb += -2.03413 
        else:
            raise RuntimeError("Unknown raceEthnicity in EducationPrevalenceModel.")
 
        if smokingStatus==SmokingStatus.NEVER:
            xb += 0 
        elif smokingStatus==SmokingStatus.FORMER:
            xb += -(-0.27219)
        elif smokingStatus==SmokingStatus.CURRENT:
            xb += -(-1.07157)
        else:
            raise RuntimeError("Unknown smokingStatus in EducationPrevalenceModel.")

        xb += -(-0.01413)*age

        lps = (xb-2.2699, xb-0.9689, xb+0.3534, xb+1.8901)
        return lps 

    def estimate_next_risk(self, person):
        linearPredictors = self.calc_linear_predictor_for_patient_characteristics(person._raceEthnicity, person._smokingStatus, person._age[-1])
        #cumulative probabilities for the first 4 education levels
        [cp1, cp2, cp3, cp4] = list(map(lambda x: self.inv_logit(x), linearPredictors))
        draw = person._rng.uniform()
        if draw<cp1:
            return Education.LESSTHANHIGHSCHOOL
        elif draw<cp2:
            return Education.SOMEHIGHSCHOOL
        elif draw<cp3:
            return Education.HIGHSCHOOLGRADUATE
        elif draw<cp4:
            return Education.SOMECOLLEGE
        elif draw<=1.:
             return Education.COLLEGEGRADUATE
        else:
             raise RuntimeError("Draw not consistent with cumulative probabilities in EducationPrevalenceModel.estimate_next_risk.")

    def inv_logit(self, lp):
        # note: limit the calculation to avoid over/under-flow issues
        if lp<-10:
            risk = 0.
        elif lp>10.:
            risk = 1.
        else:
            risk = 1/(1+np.exp(-lp))
        return risk
                                                     
