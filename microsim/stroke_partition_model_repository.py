from microsim.stroke_partition_model import (StrokePartitionModel, StrokePartitionModelFor1bpMedsAdded, StrokePartitionModelFor2bpMedsAdded,
                                             StrokePartitionModelFor3bpMedsAdded, StrokePartitionModelFor4bpMedsAdded)
from microsim.treatment import TreatmentStrategiesType

class StrokePartitionModelRepository:
    def __init__(self):
        self._models = {"0bpMedsAdded": StrokePartitionModel(),
                        "1bpMedsAdded": StrokePartitionModelFor1bpMedsAdded(),
                        "2bpMedsAdded": StrokePartitionModelFor2bpMedsAdded(),
                        "3bpMedsAdded": StrokePartitionModelFor3bpMedsAdded(),
                        "4bpMedsAdded": StrokePartitionModelFor4bpMedsAdded()}
    
    def select_outcome_model_for_person(self, person):
        tst = TreatmentStrategiesType.BP.value
        if "bpMedsAdded" in person._treatmentStrategies[tst]:
            ts = f"{person._treatmentStrategies[tst]['bpMedsAdded']}bpMedsAdded"
        else:
            ts = "0bpMedsAdded"
        return self._models[ts]
