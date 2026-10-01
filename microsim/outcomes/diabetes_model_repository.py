from microsim.outcomes.diabetes_model import DiabetesModel, DiabetesPrevalenceModel


class DiabetesModelRepository:
    def __init__(self):
        self._model = DiabetesModel()

    def select_outcome_model_for_person(self, person):
        return self._model


class DiabetesPrevalenceModelRepository:
    def __init__(self, riskScaling=1.0):
        self._model = DiabetesPrevalenceModel(riskScaling=riskScaling)

    def select_outcome_model_for_person(self, person):
        return self._model
