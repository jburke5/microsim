from microsim.wmh_model import WMHModel

class WMHModelRepository:
    """White Matter Hypodensity model repository."""
    def __init__(self):
        self._model = WMHModel()    
    
    def select_outcome_model_for_person(self, person):
        return self._model
