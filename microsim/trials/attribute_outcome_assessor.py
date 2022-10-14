from enum import Enum
import pandas as pd

from microsim.outcome import OutcomeType

class AttributeOutcomeAssessor:
    def __init__(self, attributeName, assessmentMethod):
        self.attributeName = attributeName
        self.assessmentMethod = assessmentMethod
    
    def get_outcome(self, person):
        outcome = None
        attr = getattr(person, self.attributeName)
        if self.assessmentMethod == AssessmentMethod.LAST:
            outcome = attr[-1]
        elif self.AssessmentMethod == AssessmentMethod.MEAN:
            outcome = pd.Series(attr).mean()
        else:
            raise ValueError(f"Invalid assesmment method: {self.assessmentMethod}")
        return outcome

class AssessmentMethod(Enum):
    LAST = "last"
    MEAN = "mean"

