from microsim.outcomes.epilepsy_model import EpilepsyIncidenceModel, EpilepsyPrevalenceModel


class EpilepsyModelRepository:
    def __init__(self, riskScaling=1.0):
        self._model = EpilepsyIncidenceModel(riskScaling=riskScaling)

    def select_outcome_model_for_person(self, person):
        return self._model


class EpilepsyPrevalenceModelRepository:
    def __init__(self, riskScaling=1.0):
        self._model = EpilepsyPrevalenceModel(riskScaling=riskScaling)

    def select_outcome_model_for_person(self, person):
        return self._model
