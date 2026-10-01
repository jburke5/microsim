from microsim.outcomes.non_cv_death_model import NonCVDeathModel


class NonCVModelRepository:
    def __init__(self, wmhSpecific=True, riskScaling=1.0):
        self._model = NonCVDeathModel(wmhSpecific=wmhSpecific, riskScaling=riskScaling)

    def select_outcome_model_for_person(self, person):
        return self._model
