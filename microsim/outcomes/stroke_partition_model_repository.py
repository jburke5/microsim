from microsim.outcomes.stroke_partition_model import *


class StrokePartitionModelRepository:
    def __init__(self):
        self._model = StrokePartitionModel()

    def select_outcome_model_for_person(self, person):
        return self._model


class StrokePrevalenceModelRepository:
    def __init__(self, riskScaling=1.0):
        self._model = StrokePrevalenceModel(riskScaling=riskScaling)

    def select_outcome_model_for_person(self, person):
        return self._model
