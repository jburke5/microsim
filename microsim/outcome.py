from enum import Enum


class Outcome:
    def __init__(self, type, fatal, **kwargs):
        self.type = type
        self.fatal = fatal
        self.properties = {**kwargs}

    def __repr__(self):
        return f"Outcome type: {self.type}, fatal: {self.fatal}"

    def __eq__(self, other):
        if not isinstance(other, Outcome):
            return NotImplemented
        return (self.type == other.type) and self.fatal == other.fatal


# not all outcomes are equal...some outcomes depend on other outcomes
# maybe define 2 outcome levels, base/fundamental and outcome functions that are at a higher level
# but also, different outcomes have different data structures/types, others are time dependent numbers (gcp, qalys)
# others are discrete events (mi, stroke, death)
# for now I will not work on making a taxonomy of outcomes, but simply organizing in a simple way the outcomes of interest 
# so for now, only the sequence of these outcometypes is important

class OutcomeType(Enum):
    GLOBAL_COGNITIVE_PERFORMANCE = "gcp"
    CARDIOVASCULAR = "cv"
    STROKE = "stroke"
    MI = "mi"
    NONCARDIOVASCULAR = "noncv"
    DEMENTIA = "dementia"
    DEATH = "death"
    QUALITYADJUSTED_LIFE_YEARS = "qalys"
    #making the order explicit here because some outcomes depend on other ones
    _order_ = ["GLOBAL_COGNITIVE_PERFORMANCE", "CARDIOVASCULAR", "STROKE", "MI", "NONCARDIOVASCULAR",  
               "DEMENTIA", "DEATH",  "QUALITYADJUSTED_LIFE_YEARS"]
