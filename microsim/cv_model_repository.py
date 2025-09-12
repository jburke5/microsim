from microsim.cv_model import *
from microsim.gender import NHANESGender
from microsim.treatment import TreatmentStrategiesType

class CVModelRepository:
    def __init__(self, wmhSpecific=True):
        self._models = {"male": CVModelMale(wmhSpecific=wmhSpecific),
                        "female": CVModelFemale(wmhSpecific=wmhSpecific)}

    def select_outcome_model_for_person(self, person):
        gender = "male" if person._gender==NHANESGender.MALE else "female"
        return self._models[gender]


