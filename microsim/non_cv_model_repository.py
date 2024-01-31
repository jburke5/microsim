from microsim.non_cv_death_model import NonCVDeathModel

class NonCVModelRepository:
    def __init__(self):
        self._model = NonCVDeathModel()
        
    def select_outcome_model_for_person(self, person):
        return self._model
