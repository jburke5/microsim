from microsim.ci_model import CIModel

class CIModelRepository:
    def __init__(self):
        self._model = CIModel()

    def select_outcome_model_for_person(self, person):
        return self._model 
