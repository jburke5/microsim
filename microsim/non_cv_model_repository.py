from microsim.non_cv_death_model import NonCVDeathModel

class NonCVModelRepository:
    def __init__(self, wmhSpecific=True):
        self._model = NonCVDeathModel(wmhSpecific=wmhSpecific)
        
    def select_outcome_model_for_person(self, person):
        return self._model
