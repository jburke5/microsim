from enum import Enum

#Q: should treatments be defined using their disease (eg hypertension), 
#   their effect (eg antihypertension), their drug class (eg statin), or what risk factors they affect (eg bp)?
#   treatment class, can affect outcome or risk factors
#   Do we have enough treatments in mind for how to organize these?

class DefaultTreatmentsType(Enum):
    STATIN = "statin"
    ANTI_HYPERTENSIVE_COUNT = "antiHypertensiveCount"
    OTHER_LIPID_LOWERING_MEDICATION_COUNT = "otherLipidLoweringMedicationCount"
 
class CategoricalDefaultTreatmentsType(Enum):
    STATIN = "statin"

class ContinuousDefaultTreatmentsType(Enum):
    ANTI_HYPERTENSIVE_COUNT = "antiHypertensiveCount"   

class TreatmentStrategiesType(Enum):
    BP = "bp"
    STATIN = "statin"

class TreatmentStrategyStatus(Enum):
    BEGIN = "begin"
    MAINTAIN = "maintain"
    END = "end"

class ContinuousTreatmentStrategiesType(Enum):
    pass
    #BP_MEDS_ADDED = "bpMedsAdded" #found in BP TreatmentStrategiesType

class CategoricalTreatmentStrategiesType(Enum):
    BP_MEDS_ADDED = "bpMedsAdded" #found in BP TreatmentStrategiesType
    STATINS_ADDED = "statinsAdded" 
