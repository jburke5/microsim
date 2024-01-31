from enum import Enum

class DefaultTreatmentsType(Enum):
    STATIN = "statin"
    ANTI_HYPERTENSIVE_COUNT = "antiHypertensiveCount"
    OTHER_LIPID_LOWERING_MEDICATION_COUNT = "otherLipidLoweringMedicationCount"
    
class TreatmentStrategiesType(Enum):
    BP = "bpMedsAdded"
