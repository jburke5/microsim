from enum import Enum


class TreatmentStrategiesType(Enum):
    BP = "bp"
    STATIN = "statin"
    WMD15 = "wmd15"
    WMD20 = "wmd20"
    WMD25 = "wmd25"


class TreatmentStrategyStatus(Enum):
    BEGIN = "begin"
    MAINTAIN = "maintain"
    END = "end"


class ContinuousTreatmentStrategiesType(Enum):
    pass
    # BP_MEDS_ADDED = "bpMedsAdded" #found in BP TreatmentStrategiesType


class CategoricalTreatmentStrategiesType(Enum):
    BP_MEDS_ADDED = "bpMedsAdded"  # found in BP TreatmentStrategiesType
    STATIN_MEDS_ADDED = "statinMedsAdded"
    WMD15_MEDS_ADDED = "wmd15MedsAdded"
    WMD20_MEDS_ADDED = "wmd20MedsAdded"
    WMD25_MEDS_ADDED = "wmd25MedsAdded"
