from microsim.outcomes.cognition_model import GCPModel, GCPStrokeModel, CognitionPrevalenceModel
from microsim.outcomes.outcome import OutcomeType


class CognitionModelRepository:
    def __init__(self):
        # Q: why does the GCPModel initialize an outcome model repository?
        self._models = {"gcp": GCPModel(), "gcpStroke": GCPStrokeModel()}

    def select_outcome_model_for_person(self, person):
        # we are interested in strokes that occured during the simulation, not so much on strokes that NHANES had registered for people
        # so we are selecting the gcp stroke model only when there is a stroke during the simulation
        # plus, the gcp stroke model requires quantities that we do not have from NHANES (we would need to come up with estimates)
        if person.has_outcome_during_simulation(OutcomeType.STROKE):
            return self._models["gcpStroke"]
        else:
            return self._models["gcp"]


class CognitionPrevalenceModelRepository:
    def __init__(self):
        self._model = CognitionPrevalenceModel()

    def select_outcome_model_for_person(self, person):
        return self._model
