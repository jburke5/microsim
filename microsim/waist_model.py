import numpy as np

from microsim.gender import NHANESGender

class WaistPrevalenceModel:

    def __init__(self):
        pass

    def calc_linear_predictor_for_patient_characteristics(self, age, gender, bmi):

        xb = 15.31
        xb += age*(0.14)
        xb += bmi*(3.26)
        xb += bmi*bmi*(-0.0163)
        if gender == NHANESGender.FEMALE:
            xb += -5.74
        return xb

    def estimate_next_risk(self, person):
        lp = self.calc_linear_predictor_for_patient_characteristics(person._age[-1], person._gender, person._bmi[-1])
        draw = person._rng.normal(loc = 0.0, scale = 5.85)
        return lp+draw
        
