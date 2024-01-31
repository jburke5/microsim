from microsim.cv_model import CVModelMale, CVModelFemale
from microsim.gender import NHANESGender

class CVModelRepository:
    def __init__(self):
        self._models = {"male": CVModelMale(),
                        "female": CVModelFemale()}

    def select_outcome_model_for_person(self, person):
        return self._models["male"] if person._gender==NHANESGender.MALE else self._models["female"]
