#from microsim.stroke_partition_model import (StrokePartitionModel, StrokePartitionModelFor1bpMedsAdded, StrokePartitionModelFor2bpMedsAdded,
#                                             StrokePartitionModelFor3bpMedsAdded, StrokePartitionModelFor4bpMedsAdded)
from microsim.stroke_partition_model import *
from microsim.treatment import TreatmentStrategiesType

class StrokePartitionModelRepository:
    def __init__(self):
        self._model = StrokePartitionModel()
        #self._models = {"0bpMedsAdded": StrokePartitionModel(),
        #                "1bpMedsAdded": StrokePartitionModelFor1bpMedsAdded(),
        #                "2bpMedsAdded": StrokePartitionModelFor2bpMedsAdded(),
        #                "3bpMedsAdded": StrokePartitionModelFor3bpMedsAdded(),
        #                "4bpMedsAdded": StrokePartitionModelFor4bpMedsAdded(),
        #                "5bpMedsAdded": StrokePartitionModelFor5bpMedsAdded(),
        #                "6bpMedsAdded": StrokePartitionModelFor6bpMedsAdded(),
        #                "7bpMedsAdded": StrokePartitionModelFor7bpMedsAdded(),
        #                "8bpMedsAdded": StrokePartitionModelFor8bpMedsAdded(),
        #                "9bpMedsAdded": StrokePartitionModelFor9bpMedsAdded(),
        #                "10bpMedsAdded": StrokePartitionModelFor10bpMedsAdded()}
    
    def select_outcome_model_for_person(self, person):
        return self._model

    #def select_outcome_model_for_person(self, person):
    #    tst = TreatmentStrategiesType.BP.value
    #    if "bpMedsAdded" in person._treatmentStrategies[tst]:
    #        bpMedsAdded = person._treatmentStrategies[tst]['bpMedsAdded']
    #        if bpMedsAdded < 11:
    #            ts = f"{bpMedsAdded}bpMedsAdded"
    #        else:
    #            #print(f"bpMedsAdded={bpMedsAdded}")
    #            ts = "10bpMedsAdded"
    #    else:
    #        ts = "0bpMedsAdded"
    #    return self._models[ts]
