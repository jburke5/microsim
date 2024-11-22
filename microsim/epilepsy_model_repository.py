from microsim.epilepsy_model import EpilepsyModel

class EpilepsyModelRepository:
    def __init__(self):
        self._model = EpilepsyModel()
        
    def select_outcome_model_for_person(self, person):
        return self._model