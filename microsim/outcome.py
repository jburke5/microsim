from enum import Enum

#HOW TO add a new outcome:
#
# create a new OutcomeType, pay attention to the order of the outcomes!
# (optional) create a specific Outcome class, subclass of Outcome, eg if there is an outcome phenotype that you need to store
# create a ModelRepository for the new OutcomeType
#     must initialize the model(s) this repository will use
#     must include a select_outcome_model_for_person function that will return the appropriate model for the person
# create Model class(es) that the ModelRepository for this outcome will use
#     must include a get_next_outcome function that returns either an Outcome or None
#     must include a generate_next_outcome function that initializes and returns an Outcome of the new OutcomeType
# add an entry on the OutcomeModelRepository with the key being the new OutcomeType and the value being the new ModelRepository

class Outcome:
    def __init__(self, type, fatal, priorToSim=False, **kwargs):
        self.type = type
        self.fatal = fatal
        #priorToSim: outcome happened prior to the beginning of the simulation, could be selfReported, could be doctorReported 
        self.priorToSim = priorToSim
        self.properties = {**kwargs}

    def __repr__(self):
        return f"Outcome type: {self.type}, fatal: {self.fatal}, priorToSim: {self.priorToSim}"

    def __eq__(self, other):
        if not isinstance(other, Outcome):
            return NotImplemented
        return (self.type == other.type) and self.fatal == other.fatal and self.priorToSim == other.priorToSim

# not all outcomes are equal...some outcomes depend on other outcomes
# maybe define 2 outcome levels, base/fundamental and outcome functions that are at a higher level
# for now I will not work on making a taxonomy of outcomes, but simply organizing outcomes in a simple way
# so for now, only the sequence of these outcometypes is important:
#     cognition before stroke
#     cognition before ci
#     cv, followed by stroke, followed by mi
#     noncv after cv/stroke/mi
#     qalys last
#     death right before qalys
#     dementia after cognition

class OutcomeType(Enum):
    WMH = "wmh"
    COGNITION = "cognition"
    CI = "ci"
    CARDIOVASCULAR = "cv"
    STROKE = "stroke"
    MI = "mi"
    NONCARDIOVASCULAR = "noncv"
    DEMENTIA = "dementia"
    DEATH = "death"
    QUALITYADJUSTED_LIFE_YEARS = "qalys"
    #making the order explicit here because some outcomes depend on other ones
    _order_ = ["WMH", "COGNITION", "CI", "CARDIOVASCULAR", "STROKE", "MI", "NONCARDIOVASCULAR",  
               "DEMENTIA", "DEATH",  "QUALITYADJUSTED_LIFE_YEARS"]
