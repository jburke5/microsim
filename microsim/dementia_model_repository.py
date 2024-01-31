from microsim.dementia_model import DementiaModel

class DementiaModelRepository:
    def __init__(self):
        self._model = DementiaModel()
        
    def select_outcome_model_for_person(self, person):
        return self._model
        
