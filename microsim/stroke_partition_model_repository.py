from microsim.stroke_partition_model import StrokePartitionModel

class StrokePartitionModelRepository:
    def __init__(self):
        self._model = StrokePartitionModel()
    
    def select_outcome_model_for_person(self, person):
        return self._model
