from enum import Enum
import pandas as pd

from microsim.outcome import OutcomeType

class AttributeOutcomeAssessor:
    def __init__(self, attributeName, assessmentMethod, attributeParameter=None):
        self.attributeName = attributeName
        self.assessmentMethod = assessmentMethod
        self.attributeParameter = attributeParameter #used in some, not all, assessment methods
    
    def get_outcome(self, person):
        outcome = None
        attr = getattr(person, self.attributeName)
        if self.assessmentMethod == AssessmentMethod.LAST:
            outcome = attr[-1]
        elif self.assessmentMethod == AssessmentMethod.MEAN:
            outcome = pd.Series(attr).mean()
        elif self.assessmentMethod == AssessmentMethod.SUM:
            outcome = pd.Series(attr).sum()
        elif (self.assessmentMethod == AssessmentMethod.CHANGELESSTHAN) & (self.attributeParameter is not None): #must provide attributeParameter
            outcome = ( ( attr[-1] - attr[0] ) < self.attributeParameter ) #change = final - initial
        elif (self.assessmentMethod == AssessmentMethod.CHANGEGREATERTHAN) & (self.attributeParameter is not None): #must provide attributeParameter
            outcome = ( ( attr[-1] - attr[0] ) > self.attributeParameter ) #change = final - initial
        else:
            raise ValueError(f"Invalid assesmment method: {self.assessmentMethod}")
        return outcome

    def get_name(self):
        return f"{self.attributeName}-{self.assessmentMethod.value}"

class AssessmentMethod(Enum):
    LAST = "last"
    MEAN = "mean"
    SUM = "sum"
    CHANGELESSTHAN = "changeLessThan"
    CHANGEGREATERTHAN = "changeGreaterThan"

