from microsim.outcomes.chronic_kidney_disease_model import (
    ChronicKidneyDiseaseModel,
    ChronicKidneyDiseasePrevalenceModel,
)


class ChronicKidneyDiseaseModelRepository:
    def __init__(self):
        self._model = ChronicKidneyDiseaseModel()

    def select_outcome_model_for_person(self, person):
        return self._model


class ChronicKidneyDiseasePrevalenceModelRepository:
    def __init__(self, riskScaling=1.0):
        self._model = ChronicKidneyDiseasePrevalenceModel(riskScaling=riskScaling)

    def select_outcome_model_for_person(self, person):
        return self._model
