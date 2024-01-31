from microsim.mi_partition_model import MIPartitionModel

class MIPartitionModelRepository:
    def __init__(self):
        self._model = MIPartitionModel()
    
    def select_outcome_model_for_person(self, person):
        return self._model
