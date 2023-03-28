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
        elif (self.assessmentMethod == AssessmentMethod.INCREMENTALCHANGEGREATERTHAN) & (self.attributeParameter is not None): #must provide attributeParameter
            outcome = False
            for attIndex in range(1, len(attr)):
                if attr[attIndex] - attr[attIndex-1] > self.attributeParameter:  #change = final - initial
                    return True
        elif (self.assessmentMethod == AssessmentMethod.INCREMENTALCHANGELESSTHAN) & (self.attributeParameter is not None): #must provide attributeParameter
            outcome = False
            for attIndex in range(1, len(attr)):
                if attr[attIndex] - attr[attIndex-1] < self.attributeParameter:  #change = final - initial
                    return True
        else:
            raise ValueError(f"Invalid assesmment method: {self.assessmentMethod}")
        return outcome
    
    def get_outcome_time(self, person):
        outcomeTime = len(person._age) # defaults to the last observation (i.e. never occurs)
        attr = getattr(person, self.attributeName)
        if (self.assessmentMethod == AssessmentMethod.INCREMENTALCHANGEGREATERTHAN) & (self.attributeParameter is not None): #must provide attributeParameter
            for attIndex in range(1, len(attr)):
                if attr[attIndex] - attr[attIndex-1] > self.attributeParameter:  #change = final - initial
                    return attIndex
        elif (self.assessmentMethod == AssessmentMethod.INCREMENTALCHANGELESSTHAN) & (self.attributeParameter is not None): #must provide attributeParameter
            for attIndex in range(1, len(attr)):
                if attr[attIndex] - attr[attIndex-1] < self.attributeParameter:  #change = final - initial
                    return attIndex
        else:
            raise ValueError(f"Invalid assesmment method: {self.assessmentMethod}")
        
        return outcomeTime

    def get_name(self):
        return f"{self.attributeName}-{self.assessmentMethod.value}"

class AssessmentMethod(Enum):
    LAST = "last"
    MEAN = "mean"
    SUM = "sum"
    CHANGELESSTHAN = "changeLessThan"
    CHANGEGREATERTHAN = "changeGreaterThan"
    INCREMENTALCHANGEGREATERTHAN = "incrementalChangeGreaterThan" # captures whether there is EVER a change greater than a threshold
    INCREMENTALCHANGELESSTHAN = "incrementalChangeLessThan" # captures where there is EVER a change less than a threshold

