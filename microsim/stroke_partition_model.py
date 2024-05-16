from microsim.statsmodel_linear_risk_factor_model import StatsModelLinearRiskFactorModel
from microsim.regression_model import RegressionModel
from microsim.data_loader import load_model_spec
from microsim.outcome import OutcomeType, Outcome
from microsim.outcome_details.stroke_details import StrokeSubtypeModelRepository, StrokeNihssModel, StrokeTypeModel
from microsim.stroke_outcome import StrokeOutcome, StrokeSubtype, StrokeType, Localization

import scipy.special as scipySpecial

# there are 2 approaches that can be taken with partition models
# 1: this model asks the CVModel to see if there was an cv outcome for this person, and that outcome is just used without being stored
#    on the Person-instance
# 2: the CVModel is asked before StrokePartitionModel is called and the Person-object stores the CV outcome, and the strokeModel
#.  is just checking on the person object to see if there was a cv outcome in the current year
#   We now use the 2nd approach, and we will double check that the outcomes are called in the correct order.

class StrokePartitionModel(StatsModelLinearRiskFactorModel):
    """Fatal stroke probability estimated from our meta-analysis of BASIC, NoMAS, GCNKSS, REGARDS."""
    interceptChangeFor1bpMedsAdded = -0.1134942

    def __init__(self, intercept=None):
        model_spec = load_model_spec("StrokeMIPartitionModel")
        #the assumption for the subclasses of this class is that the intercept with 0 bpMedsAdded is -2.3109730587083006
        if intercept is not None:
            model_spec["coefficients"]["Intercept"] = intercept
        super().__init__(RegressionModel(**model_spec))
        self._stroke_case_fatality = 0.15
        self._stroke_secondary_case_fatality = 0.15
   
    def will_have_fatal_stroke(self, person):
        fatalStrokeProb = self._stroke_case_fatality
        fatalProb = self._stroke_secondary_case_fatality if person._stroke else fatalStrokeProb
        return person._rng.uniform(size=1) < fatalProb

    def get_next_stroke_probability(self, person):
        #Q: I am not sure why it was set to 0 at the beginning
        strokeProbability = 0
        strokeProbability = scipySpecial.expit( super().estimate_next_risk(person) )
        return strokeProbability
    
    def generate_next_outcome(self, person):
        fatal = self.will_have_fatal_stroke(person)
        nihss = StrokeNihssModel().estimate_next_risk(person)
        strokeSubtype = StrokeSubtypeModelRepository().get_stroke_subtype(person)
        strokeType = StrokeTypeModel().get_stroke_type(person)
        #localization = Localization.LEFT_HEMISPHERE
        #disability = 3 
        #return StrokeOutcome(fatal, nihss, strokeType, strokeSubtype, localization, disability)
        return StrokeOutcome(fatal, nihss, strokeType, strokeSubtype)
        
    def update_cv_outcome(self, person, fatal):
        person._outcomes[OutcomeType.CARDIOVASCULAR][-1][1].fatal = fatal
        
    def get_next_outcome(self, person):
        if person.has_outcome_at_current_age(OutcomeType.CARDIOVASCULAR):
            if person._rng.uniform(size=1) < self.get_next_stroke_probability(person):
                strokeOutcome = self.generate_next_outcome(person)
                #if we decide to use different stroke/mi models, double check if fatality needs to be decided in CVModel
                self.update_cv_outcome(person, strokeOutcome.fatal)
                return strokeOutcome
            else: 
                return None

class StrokePartitionModelFor1bpMedsAdded(StrokePartitionModel):
    def __init__(self):
        #intercept = -2.4295668
        intercept = -2.31097305 + 1*self.interceptChangeFor1bpMedsAdded
        super().__init__(intercept=intercept)

class StrokePartitionModelFor2bpMedsAdded(StrokePartitionModel):
    def __init__(self):
        #intercept = -2.5334375
        intercept = -2.31097305 + 2*self.interceptChangeFor1bpMedsAdded
        super().__init__(intercept=intercept)

class StrokePartitionModelFor3bpMedsAdded(StrokePartitionModel):
    def __init__(self):
        #intercept = -2.66499999
        intercept = -2.31097305 + 3*self.interceptChangeFor1bpMedsAdded
        super().__init__(intercept=intercept)

class StrokePartitionModelFor4bpMedsAdded(StrokePartitionModel):
    def __init__(self):
        #intercept = -2.764950
        intercept = -2.31097305 + 4*self.interceptChangeFor1bpMedsAdded
        super().__init__(intercept=intercept)

class StrokePartitionModelFor5bpMedsAdded(StrokePartitionModel):
    def __init__(self):
        intercept = -2.31097305 + 5*self.interceptChangeFor1bpMedsAdded
        super().__init__(intercept=intercept)

class StrokePartitionModelFor6bpMedsAdded(StrokePartitionModel):
    def __init__(self):
        intercept = -2.31097305 + 6*self.interceptChangeFor1bpMedsAdded
        super().__init__(intercept=intercept)

class StrokePartitionModelFor7bpMedsAdded(StrokePartitionModel):
    def __init__(self):
        intercept = -2.31097305 + 7*self.interceptChangeFor1bpMedsAdded
        super().__init__(intercept=intercept)

class StrokePartitionModelFor8bpMedsAdded(StrokePartitionModel):
    def __init__(self):
        intercept = -2.31097305 + 8*self.interceptChangeFor1bpMedsAdded
        super().__init__(intercept=intercept)

class StrokePartitionModelFor9bpMedsAdded(StrokePartitionModel):
    def __init__(self):
        intercept = -2.31097305 + 9*self.interceptChangeFor1bpMedsAdded
        super().__init__(intercept=intercept)

class StrokePartitionModelFor10bpMedsAdded(StrokePartitionModel):
    def __init__(self):
        intercept = -2.31097305 + 10*self.interceptChangeFor1bpMedsAdded
        super().__init__(intercept=intercept)


