from microsim.outcomes.cv_model import CVModelFemale, CVModelMale, CVPrevalenceModel
from microsim.risk_factors.gender import NHANESGender


class CVModelRepository:
    def __init__(self, wmhSpecific=True, riskScaling=1.0):
        self._models = {
            "male": CVModelMale(wmhSpecific=wmhSpecific, riskScaling=riskScaling),
            "female": CVModelFemale(wmhSpecific=wmhSpecific, riskScaling=riskScaling),
        }

    def select_outcome_model_for_person(self, person):
        gender = "male" if person._gender == NHANESGender.MALE else "female"
        return self._models[gender]


class CVPrevalenceModelRepository:
    def __init__(self, riskScaling=1.0):
        self._model = CVPrevalenceModel(riskScaling=riskScaling)

    def select_outcome_model_for_person(self, person):
        return self._model
