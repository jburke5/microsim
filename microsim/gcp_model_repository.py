from microsim.gcp_model import GCPModel
from microsim.gcp_stroke_model import GCPStrokeModel
from microsim.outcome import OutcomeType

class GCPModelRepository:
    def __init__(self):
        #Q: why does the GCPModel initialize an outcome model repository?
        self._models = {"gcp": GCPModel(),
                        "gcpStroke": GCPStrokeModel()}

    #we are interested in strokes that occured during the simulation, not so much on strokes that NHANES had registered for people
    #so we are selecting the gcp stroke model only when there is a stroke during the simulation
    #plus, the gcp stroke model requires quantities that we do not have from NHANES (we would need to come up with estimates)
        
    def select_outcome_model_for_person(self, person):
        if person.has_outcome_during_simulation(OutcomeType.STROKE):
            return self._models["gcpStroke"]  
        else:
            return self._models["gcp"]

