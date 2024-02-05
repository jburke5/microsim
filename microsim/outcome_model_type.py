from enum import Enum

#Q since everything is included in OutcomeType, I think this class can be deleted...
class OutcomeModelType(Enum):
    CARDIOVASCULAR = "cardiovascular"
    NON_CV_MORTALITY = "mortality"
    GLOBAL_COGNITIVE_PERFORMANCE = "gcp"
    DEMENTIA = "dementia"
