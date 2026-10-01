from microsim.outcomes.mci_model import MCIModel


class MCIModelRepository:
    def __init__(self, riskScaling=1.0):
        self._model = MCIModel(riskScaling=riskScaling)

    def select_outcome_model_for_person(self, person):
        return self._model
