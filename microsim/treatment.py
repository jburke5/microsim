from enum import Enum

#Q: should treatments be defined using their disease (eg hypertension), 
#their effect (eg antihypertension), their drug class (eg statin), or what risk factors they affect (eg bp)?
#treatment class, can affect outcome or risk factors

class DefaultTreatmentsType(Enum):
    STATIN = "statin"
    ANTI_HYPERTENSIVE_COUNT = "antiHypertensiveCount"
    OTHER_LIPID_LOWERING_MEDICATION_COUNT = "otherLipidLoweringMedicationCount"
    
class TreatmentStrategiesType(Enum):
    BP = "bpMedsAdded"
