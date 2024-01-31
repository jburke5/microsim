from microsim.death_model import DeathModel

class DeathModelRepository:
    def __init__(self):
        self._model = DeathModel()
        
    def select_outcome_model_for_person(self, person):
        return self._model
