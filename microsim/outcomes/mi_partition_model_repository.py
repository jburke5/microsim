from microsim.outcomes.mi_partition_model import MIPartitionModel, MIPrevalenceModel


class MIPartitionModelRepository:
    def __init__(self):
        self._model = MIPartitionModel()

    def select_outcome_model_for_person(self, person):
        return self._model


class MIPrevalenceModelRepository:
    def __init__(self):
        self._model = MIPrevalenceModel()

    def select_outcome_model_for_person(self, person):
        return self._model
